"""AWStream offline simulation driver."""
import csv
import os
import glob

from videostorm.VideoStorm import VideoStorm
from videos import get_dataset_class, KittiVideo


def get_seg_paths(data_path, dataset_name, video_name):
    """Return all segment paths in a dataset."""
    if video_name is not None and video_name:
        seg_paths = [os.path.join(data_path, video_name)]
    elif dataset_name == 'kitti':
        seg_paths = []
        for loc in KittiVideo.LOCATIONS:
            for seg_path in sorted(
                    glob.glob(os.path.join(data_path, loc, '*'))):
                if not os.path.isdir(seg_path):
                    continue
                seg_paths.append(seg_path)
    elif dataset_name == 'mot15':
        raise NotImplementedError
    elif dataset_name == 'mot16':
        raise NotImplementedError
    elif dataset_name == 'waymo':
        seg_paths = glob.glob(os.path.join(data_path, '*', 'FRONT'))
    elif dataset_name == 'youtube':
        raise NotImplementedError
    else:
        raise NotImplementedError
    return seg_paths


def run(args):
    """Run VideoStorm simulation."""
    dataset_class = get_dataset_class(args.dataset)
    seg_paths = get_seg_paths(args.data_root, args.dataset, args.video)
    original_resolution = args.original_resolution
    overfitting = args.overfitting
    model_list = args.model_list if not overfitting else ['FasterRCNN']
    sample_step_list = args.sample_step_list
    short_video_length = args.short_video_length
    profile_length = args.profile_length
    detection_path = args.detection_root
    profile_filename = args.profile_filename
    output_filename = args.output_filename

    if overfitting:
        assert short_video_length == profile_length, "short_video_length " \
            "should equal to profile_length when overfitting."
    else:
        assert short_video_length >= profile_length, "short_video_length " \
            "should no less than profile_length."

    pipeline = VideoStorm(sample_step_list, model_list, profile_filename)
    with open(output_filename, 'w', 1) as f_out:
        writer = csv.writer(f_out)
        writer.writerow(
            ["video_name", 'model', 'gpu time', "frame_rate", "f1"])
        for seg_path in seg_paths:
            print(seg_path)
            seg_name = os.path.basename(seg_path)
            # if seg_name + '.mp4' in saved_videos:
            #     print('jump over', seg_name)
            #     continue
            # loading videos
            img_path = os.path.join(seg_path, 'FRONT')
            dt_file = os.path.join(
                detection_path, seg_name, 'FRONT', 'profile',
                f"updated_gt_FasterRCNN_COCO_no_filter_{original_resolution}.csv")
            original_video = dataset_class(
                seg_path, seg_name, original_resolution, dt_file, img_path,
                filter_flag=True)
            videos = {}
            for model in model_list:
                img_path = os.path.join(seg_path, 'FRONT')
                dt_file = os.path.join(
                    detection_path, seg_name, 'FRONT', 'profile',
                    f'updated_gt_{model}_COCO_no_filter_{original_resolution}.csv')
                video = dataset_class(seg_name, original_resolution, dt_file,
                                      img_path, filter_flag=True)
                videos[model] = video
                print('loading {}...'.format(dt_file))

            # compute number of short videos can be splitted
            if original_video.duration > short_video_length:
                nb_short_videos = original_video.frame_count // (
                    short_video_length*original_video.frame_rate)
            else:
                nb_short_videos = 1

            for i in range(nb_short_videos):
                clip = seg_name + '_' + str(i)
                start_frame = i * short_video_length * \
                    original_video.frame_rate + \
                    original_video.start_frame_index

                end_frame = min((i + 1) * short_video_length *
                                original_video.frame_rate,
                                original_video.end_frame_index)
                print('{} start={} end={}'.format(
                    clip, start_frame, end_frame))
                # use 30 seconds video for profiling
                if overfitting:
                    profile_start = start_frame
                    profile_end = end_frame
                else:
                    profile_start = start_frame
                    profile_end = start_frame + original_video.frame_rate * \
                        profile_length - 1

                print('profile {} start={} end={}'.format(
                    clip, profile_start, profile_end))
                best_frame_rate, best_model = pipeline.profile(
                    clip, videos, original_video, [profile_start, profile_end])
                best_sample_rate = original_video.frame_rate/best_frame_rate

                if overfitting:
                    test_start = start_frame
                    test_end = end_frame
                else:
                    test_start = profile_end + 1
                    test_end = end_frame

                print('Evaluate {} start={} end={}'.format(
                    clip, test_start, test_end))
                f1_score, relative_gpu_time, triggered_frames_tmp = \
                    pipeline.evaluate(videos[best_model], original_video,
                                      best_sample_rate, [test_start, test_end])

                print(clip, best_model, relative_gpu_time,
                      best_frame_rate / original_video.frame_rate, f1_score)
                writer.writerow(
                    [clip, best_model, relative_gpu_time, f1_score])
