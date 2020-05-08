"""AWStream offline simulation driver."""
import csv
import os
import glob

from awstream.Awstream import Awstream
from videos import KittiVideo, MOT15Video, MOT16Video, WaymoVideo, YoutubeVideo


def get_dataset_class(dataset_name):
    """Return the class template with respect to dataset name."""
    if dataset_name == 'kitti':
        return KittiVideo
    elif dataset_name == 'mot15':
        return MOT15Video
    elif dataset_name == 'mot16':
        return MOT16Video
    elif dataset_name == 'waymo':
        return WaymoVideo
    elif dataset_name == 'youtube':
        return YoutubeVideo
    else:
        raise NotImplementedError


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
    """Run AWStream simulation."""
    dataset_class = get_dataset_class(args.dataset)
    seg_paths = get_seg_paths(args.data_root, args.dataset, args.video)
    original_resolution = args.resolution
    resolution_list = args.resolution_list
    sample_step_list = args.sample_step_list if not args.overfitting else [1]
    qp_list = args.quality_parameter_list
    short_video_length = args.short_video_length
    profile_length = args.profile_length
    detection_path = args.detection_root
    output_path = args.output_path

    if args.overfitting:
        assert short_video_length == profile_length, "short_video_length " \
            "should equal to profile_length when overfitting."
    else:
        assert short_video_length >= profile_length, "short_video_length " \
            "should no less than profile_length."

    pipeline = Awstream(sample_step_list, resolution_list, qp_list,
                        args.profile_filename, args.video_save_path)
    with open(args.output_filename, 'w', 1) as f_out:
        writer = csv.writer(f_out)
        writer.writerow(['dataset', 'best_resolution',
                         'f1', 'frame_rate', 'bandwidth'])
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
                f"updated_gt_FasterRCNN_COCO_no_filter_{args.original_resolution}.csv")
            original_video = dataset_class(
                seg_path, seg_name, original_resolution, dt_file, img_path,
                filter_flag=True)
            videos = {}
            for resol in resolution_list:
                img_path = os.path.join(seg_path, 'FRONT')
                dt_file = os.path.join(
                    detection_path, seg_name, 'FRONT', 'profile',
                    f'updated_gt_FasterRCNN_COCO_no_filter_{resol}.csv')
                video = WaymoVideo(seg_name, resol, dt_file,
                                   img_path, filter_flag=True)
                videos[resol] = video
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
                if args.overfitting:
                    profile_start = start_frame
                    profile_end = end_frame
                else:
                    profile_start = start_frame
                    profile_end = start_frame + original_video.frame_rate * \
                        args.profile_length - 1

                print('profile {} start={} end={}'.format(
                    clip, profile_start, profile_end))
                best_resol, best_fps, best_bw = \
                    pipeline.profile(clip, videos, original_video,
                                     [profile_start, profile_end])

                print("Profile {}: best resol={}, best fps={}, best bw={}"
                      .format(clip, best_resol, best_fps, best_bw))

                if args.overfitting:
                    test_start = start_frame
                    test_end = end_frame
                else:
                    test_start = profile_end + 1
                    test_end = end_frame

                print('Evaluate {} start={} end={}'.format(
                    clip, test_start, test_end))
                f1_score, relative_bw = pipeline.evaluate(
                    os.path.join(output_path, clip + '.mp4'),
                    original_video, videos[str(best_resol[1])+'p'], best_fps,
                    [test_start, test_end])

                print('{} best fps={}, best resolution={} ==> tested f1={}'
                      .format(clip, best_fps/original_video.frame_rate,
                              best_resol, f1_score))
                writer.writerow([clip, str(best_resol[1]) + 'p', f1_score,
                                 best_fps/original_video.frame_rate,
                                 relative_bw])
