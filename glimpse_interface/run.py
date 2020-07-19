"""Glimpse offline simulation driver."""
import csv
import os

from glimpse_interface.Glimpse import Glimpse, Glimpse_Temporal, Glimpse_Model
from videos import get_dataset_class, get_seg_paths
from utils.utils import load_COCOlabelmap


def run(args):
    """Run Glimpse simulation."""
    dataset_class = get_dataset_class(args.dataset)
    seg_paths = get_seg_paths(args.data_root, args.dataset, args.video)
    original_resolution = args.original_resolution
    coco_id2name, coco_name2id = load_COCOlabelmap(args.coco_label_file)
    classes_interested = {coco_name2id[class_type]
                          for class_type in set(args.classes_interested)}
    overfitting = args.overfitting
    tracking_error_threshold_list = args.tracking_error_threshold_list
    frame_difference_divisor_list = \
        args.frame_difference_threshold_divisor_list
    short_video_length = args.short_video_length
    profile_length = args.profile_length
    # detection_path = args.detection_root
    profile_filename = args.profile_filename
    output_filename = args.output_filename
    vis_video = args.vis_video
    output_video = args.output_video

    if overfitting:
        assert short_video_length == profile_length, "short_video_length " \
            "should equal to profile_length when overfitting."
    else:
        assert short_video_length >= profile_length, "short_video_length " \
            "should no less than profile_length."

    glimpse_temporal = Glimpse_Temporal(1, 5, 1280*720//2, 10)
    glimpse_model = Glimpse_Model(1280*720//2, 10)

    pipeline = Glimpse(glimpse_temporal, glimpse_model)
    with open(profile_filename, 'w', 1) as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["video_name", "f1", "bw"])
        
        for seg_path in seg_paths:
             # compute number of short videos can be splitted
            seg_name = os.path.basename(seg_path)
            # video = dataset_class(
            #             seg_path, seg_name, original_resolution,
            #             'faster_rcnn_resnet101', filter_flag=True,
            #             classes_interested=classes_interested)
            video = dataset_class(
                        seg_path, seg_name, original_resolution,
                        'faster_rcnn_resnet101', filter_flag=False)
            nb_short_videos = 0
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
                _, triggered_frame, f1 = pipeline.run(video, profile_start, profile_end, output_filename, vis_video=vis_video, output_video=output_video)
                bw = 1.0*triggered_frame/(profile_end-profile_start + 1)
                writer.writerow([clip, f1, bw])

        