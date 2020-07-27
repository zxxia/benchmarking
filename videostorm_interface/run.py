import csv
import os

from utils.utils import load_COCOlabelmap
from videos import get_dataset_class, get_seg_paths
from videostorm_interface.VideoStorm import VideoStorm, VideoStorm_Temporal, VideoStorm_Spacial, VideoStorm_Model

def run(args):
    """Run VideoStorm simulation."""
    dataset_class = get_dataset_class(args.dataset)
    seg_paths = get_seg_paths(args.data_root, args.dataset, args.video)
    original_resolution = args.original_resolution
    overfitting = args.overfitting
    model_list = args.module_list if not overfitting else ['faster_rcnn_resnet101']
    sample_step_list = args.sample_step_list
    short_video_length = args.short_video_length
    profile_length = args.profile_length
    profile_filename = args.profile_filename
    output_filename = args.output_filename

    qp_list = args.quality_parameter_list
    output_filename = args.output_filename

    videostorm_temporal_flag = args.videostorm_temporal_flag
    videostorm_spacial_flag = args.videostorm_spacial_flag
    videostorm_model_flag = args.videostorm_model_flag
    spacial_resolution = args.spacial_resolution

    coco_id2name, coco_name2id = load_COCOlabelmap(args.coco_label_file)
    classes_interested = {coco_name2id[class_type] for class_type in args.classes_interested}
    if overfitting:
        assert short_video_length == profile_length, "short_video_length should equal to profile_length when overfitting."
    else:
        assert short_video_length >= profile_length, "short_video_length should no less than profile_length."

    # videostorm_temporal = VideoStorm_Temporal(sample_step_list, )
    # videostorm_spacial = VideoStorm_Spacial(original_resolution, dataset_class)
    # videostorm_model = VideoStorm_Model(model_list)
    # print("INPUT MODEL_LIST!!!!!!!!!!!!!!!!!!:", model_list)
    pipeline = VideoStorm(sample_step_list, model_list, original_resolution, spacial_resolution, qp_list, profile_filename, output_path, videostorm_temporal_flag, videostorm_spacial_flag, videostorm_model_flag)

    with open(output_filename, 'w', 1) as f_out:
        writer = csv.writer(f_out)
        writer.writerow(['video_name', 'model', 'gpu time', 'frame_rate', 'f1'])
        for seg_path in seg_paths:
            seg_name = os.path.basename(seg_path)
            original_video = dataset_class(seg_path, seg_name, original_resolution, 'faster_rcnn_resnet101', filter_flag=True, classes_interested=classes_interested)
            videos = {}
            print("")
            #print("MODEL_LIST!!!!!!!!!!!!!!!!!!!!!:", pipeline.videostorm_model.model_list)
            for model in pipeline.videostorm_model.model_list:
                print(pipeline.videostorm_spacial.resolution, original_resolution)
                video = dataset_class(seg_path, seg_name, pipeline.videostorm_spacial.resolution, model, filter_flag=True, classes_interested=classes_interested)
                videos[model] = video

            if original_video.duration > short_video_length:
                nb_short_videos = original_video.frame_count // (short_video_length * original_video.frame_rate)
            else:
                nb_short_videos = 1

            # equals: while seg = Next_segment(video):
            for i in range(nb_short_videos):
                clip = seg_name + '_' + str(i)
                start_frame = i * short_video_length * original_video.frame_rate + original_video.start_frame_index
                end_frame = min((i + 1) * short_video_length * original_video.frame_rate, original_video.end_frame_index)
                print('{} start={} end={}'.format(clip, start_frame, end_frame))

                if overfitting:
                    profile_start = start_frame
                    profile_end = end_frame
                else:
                    profile_start = start_frame
                    profile_end = start_frame + original_video.frame_rate * profile_length - 1
                print('profile {} start={} end={}'.format(clip, profile_start, profile_end))
                '''
                print('Check params clip: ', clip, videos, original_video, [profile_start, profile_end])
                print('Check params videos: ', videos)
                print('Check params original_video: ', original_video)
                print('Check params range: ',[profile_start, profile_end])
                '''
                best_frame_rate, best_model = pipeline.Server(clip, videos, original_video, [profile_start, profile_end])
                # 输入参数的是分母
                best_sample_rate = original_video.frame_rate/best_frame_rate

                if overfitting:
                    test_start = start_frame
                    test_end = end_frame
                else:
                    test_start = profile_end + 1
                    test_end = end_frame

                print('Evaluate {} start={} end={}'.format(clip, test_start, test_end))
                f1_score, relative_gpu_time, relative_bandwith = pipeline.evaluate(clip, videos[best_model], original_video, best_frame_rate, [test_start, test_end])
                print('Evaluate each frame result: clip={}, best_model={}, relative_gpu_time={}, best_frame_rate / original_video.frame_rate={}, f1_score={}'.format(clip, best_model, relative_gpu_time, best_frame_rate / original_video.frame_rate, f1_score))
                writer.writerow([clip, best_model, relative_gpu_time, best_frame_rate / original_video.frame_rate, f1_score])




