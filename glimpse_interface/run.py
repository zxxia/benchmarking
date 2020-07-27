"""Glimpse offline simulation driver."""
import csv
import os

from glimpse_interface.Glimpse import Glimpse, Glimpse_Temporal, Glimpse_Model
from videos import get_dataset_class, get_seg_paths
from utils.utils import load_COCOlabelmap
from evaluation.f1 import evaluate_frame, compute_f1


def run(args):
    """Run Glimpse simulation."""
    dataset_class = get_dataset_class(args.dataset)
    seg_paths = get_seg_paths(args.data_root, args.dataset, args.video)
    original_resolution = args.original_resolution
    coco_id2name, coco_name2id = load_COCOlabelmap(args.coco_label_file)
    classes_interested = {coco_name2id[class_type]
                          for class_type in set(args.classes_interested)}
    overfitting = args.overfitting
    tracking_error_threshold = args.tracking_error_threshold
    frame_difference_divisor = \
        args.frame_difference_threshold_divisor
    short_video_length = args.short_video_length
    profile_length = args.profile_length
    # detection_path = args.detection_root
    profile_filename = args.profile_filename
    output_filename = args.output_filename
    vis_video = args.vis_video
    output_video = args.output_video
    cache_size = args.cache_size
    sample_rate = args.sample_rate
    tracking_method = args.tracking_method

    if overfitting:
        assert short_video_length == profile_length, "short_video_length " \
            "should equal to profile_length when overfitting."
    else:
        assert short_video_length >= profile_length, "short_video_length " \
            "should no less than profile_length."
    
    Config = {"cache_size":cache_size, "sample_rate":sample_rate, "tracking_method":tracking_method,
            "tracking_error_thres":tracking_error_threshold, "frame_diff_thres":0}

    glimpse_temporal = Glimpse_Temporal(Config)
    glimpse_model = Glimpse_Model(Config)

    pipeline = Glimpse(glimpse_temporal, glimpse_model)
    with open(profile_filename+"_"+tracking_method+"_"+str(cache_size)+".csv", 'w', 1) as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["video_name", "f1", "bw", "gpu"])
        
        for seg_path in seg_paths:
             # compute number of short videos can be splitted
            seg_name = os.path.basename(seg_path)
            video = dataset_class(
                         seg_path, seg_name, original_resolution,
                         'faster_rcnn_resnet101', filter_flag=True,
                         classes_interested=classes_interested)
            nb_short_videos = 0
            Config["frame_diff_thres"] = (video.resolution[0]*video.resolution[1])//frame_difference_divisor
            if video.duration > short_video_length:
                nb_short_videos = video.frame_count // (
                    short_video_length*video.frame_rate)
            else:
                nb_short_videos = 1
            print(nb_short_videos)
            for i in range(nb_short_videos):
                clip = seg_name + '_' + str(i)
                start_frame = i * short_video_length * \
                    video.frame_rate + video.start_frame_index

                end_frame = min((i + 1) * short_video_length *
                                video.frame_rate, video.end_frame_index)
                print('{} start={} end={}'.format(
                    clip, start_frame, end_frame))
                # use 30 seconds video for profiling
                if overfitting:
                    profile_start = start_frame
                    profile_end = end_frame
                else:
                    profile_start = start_frame
                    profile_end = start_frame + video.frame_rate * \
                        profile_length - 1

                print('profile {} start={} end={}'.format(
                    clip, profile_start, profile_end))
                Seg = {"video":video, "start_frame":start_frame, "end_frame":end_frame}
                _, decision_results, detection_results = pipeline.run(Seg, Config)
                with open(output_filename, 'a', 1) as o_out:
                    result_writer = csv.writer(o_out)
                    result_writer.writerow(['frame id', 'xmin', 'ymin', 'xmax', 
                            'ymax', 'class' ,' score', 'object id'])
                    yellow = (0, 255, 255)
                    videoWriter = None
                    if vis_video and output_video is not None:
                        fps = 30
                        size = (video.resolution[0],video.resolution[1])
                        fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
                        videoWriter = cv2.VideoWriter(output_video, fourcc, fps, size) 
                    print("saving results ...") 
                    total_tp, total_fp, total_fn = 0,0,0
                    for i, result in enumerate(detection_results):
                        tp, fp, fn = evaluate_frame(video.get_frame_detection(start_frame + i), result)
                        total_tp += tp
                        total_fp += fp
                        total_fn += fn
                        if vis_video and output_video is not None:
                            frame = video.get_frame_image(start_frame + i)
                        for box in result:
                            result_writer.writerow([i+1] + box)
                            if vis_video and output_video is not None:
                                cv2.rectangle(frame, (int(box[0]),int(box[1])), 
                                            (int(box[2]),int(box[3])), yellow, 2)
                        if vis_video and output_video is not None:
                            videoWriter.write(frame)
                    if vis_video and output_video is not None:
                        videoWriter.release()
                f1_score = compute_f1(total_tp, total_fp, total_fn)
                origin_filesize = 0
                filesize = 0
                frame_num = 0
                original_frame_num = 0
                for i in range(profile_start, profile_end + 1):
                    origin_filesize += video.get_frame_filesize(i)
                    original_frame_num += 1
                    if i in decision_results["Frame_Decision"]:
                        if not decision_results["Frame_Decision"][i]["skip"]:
                            filesize += video.get_frame_filesize(i)
                            frame_num += 1
                bw = 1.0*filesize/origin_filesize
                gpu = 1.0*frame_num/original_frame_num
                writer.writerow([clip, f1_score, bw, gpu])

        