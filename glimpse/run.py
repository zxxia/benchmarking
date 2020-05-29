"""Glimpse offline simulation driver."""
import csv
import os

from glimpse.Glimpse import Glimpse
from videos import get_dataset_class, get_seg_paths


def run(args):
    """Run Glimpse simulation."""
    dataset_class = get_dataset_class(args.dataset)
    seg_paths = get_seg_paths(args.data_root, args.dataset, args.video)
    original_resolution = args.original_resolution
    overfitting = args.overfitting
    tracking_error_threshold_list = args.tracking_error_threshold_list
    frame_difference_divisor_list = \
        args.frame_difference_threshold_divisor_list
    short_video_length = args.short_video_length
    profile_length = args.profile_length
    # detection_path = args.detection_root
    profile_filename = args.profile_filename
    output_filename = args.output_filename

    if overfitting:
        assert short_video_length == profile_length, "short_video_length " \
            "should equal to profile_length when overfitting."
    else:
        assert short_video_length >= profile_length, "short_video_length " \
            "should no less than profile_length."

    pipeline = Glimpse(frame_difference_divisor_list,
                       tracking_error_threshold_list, profile_filename)
    with open(output_filename, 'w', 1) as f_out:
        writer = csv.writer(f_out)
        writer.writerow(
            ["video_name", 'model', 'gpu time', "frame_rate", "f1"])
        for seg_path in seg_paths:
            print(seg_path)
            seg_name = os.path.basename(seg_path)
            # loading videos
            video = dataset_class(seg_path, seg_name, original_resolution,
                                  'faster_rcnn_resnet101', filter_flag=True)

            # compute number of short videos can be splitted
            if video.duration > short_video_length:
                nb_short_videos = video.frame_count // (
                    short_video_length*video.frame_rate)
            else:
                nb_short_videos = 1

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
                best_frame_difference_threshold_divisor, \
                    best_tracking_error_threshold = pipeline.profile(
                        clip, video, profile_start, profile_end)

                if overfitting:
                    test_start = start_frame
                    test_end = end_frame
                else:
                    test_start = profile_end + 1
                    test_end = end_frame

                print('Evaluate {} start={} end={}'.format(
                    clip, test_start, test_end))
                ideal_triggered_frame, f1, trigger_f1, pix_change_obj, \
                    pix_change_bg, frame_diff_triggered, tracking_triggered, \
                    frames_log = pipeline.evaluate(
                        video, test_start, test_end,
                        best_frame_difference_threshold_divisor,
                        best_tracking_error_threshold)

                frames_triggered = frame_diff_triggered.union(
                    tracking_triggered)
                bw = 0
                for frame_idx in frames_triggered:
                    bw += video.get_frame_filesize(frame_idx)
                final_fps = len(frames_triggered) / (test_end - test_start + 1)
                frame_diff_fps = len(frame_diff_triggered) / \
                    (test_end - test_start + 1)
                tracking_fps = len(tracking_triggered) / \
                    (test_end - test_start + 1)
                writer.writerow(
                    [clip, best_frame_difference_threshold_divisor,
                     best_tracking_error_threshold, f1, final_fps,
                     frame_diff_fps, tracking_fps, bw])
