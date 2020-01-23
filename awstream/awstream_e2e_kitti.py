"""AWStream on Waymo."""
# import argparse
import csv
import glob
import os
# import pdb
from benchmarking.awstream.Awstream import Awstream
from benchmarking.video import KittiVideo


DATA_PATH = '/data/zxxia/ekya/datasets/waymo_images'
DETECTION_PATH = '/data/zxxia/ekya/datasets/waymo_images_detection_results'
# PATH = '/data/zxxia/benchmarking/results/videos/'
# PATH = '/home/zxxia/benchmarking/videos/'
# DATA_PATH = '/data2/zxxia/videos/'
# TEMPORAL_SAMPLING_LIST = [1]
TEMPORAL_SAMPLING_LIST = [20, 15, 10, 5, 4, 3, 2.5, 2, 1.8, 1.5, 1.2, 1]
OFFSET = 0
RESOLUTION_LIST = ['720p', '540p', '480p', '360p']  # '2160p', '1080p',
# RESOLUTION_LIST = ['720p']  # '2160p', '1080p',
ORIGINAL_RESOL = '720p'

PROFILE_LENGTH = 10
LOG = 'awstream_e2e_kitti_profile.csv'
OUTPUT_PATH = '/data/zxxia/KITTI/awstream_save_videos'
ROOT = '/data/zxxia/KITTI'

LOCATIONS = ['City', 'Residential', 'Road']


def main():
    """Run E2E AWStream on kitti videos."""
    # args = parse_args()
    pipeline = Awstream(TEMPORAL_SAMPLING_LIST, RESOLUTION_LIST, None, LOG)
    with open('awstream_e2e_kitti.csv', 'w', 1) as f_out:
        writer = csv.writer(f_out)
        writer.writerow(['dataset', 'best_resolution',
                         'f1', 'frame_rate', 'bandwidth'])
        for loc in LOCATIONS:
            for seg_path in sorted(glob.glob(os.path.join(ROOT, loc, '*'))):
                if not os.path.isdir(seg_path):
                    continue
                video_name = loc + '_' + os.path.basename(seg_path)
                print(seg_path)
                # loading videos
                img_path = os.path.join(
                    seg_path, 'image_02', 'data', ORIGINAL_RESOL)
                dt_file = os.path.join(
                    seg_path, 'image_02', 'data', ORIGINAL_RESOL, 'profile',
                    'updated_gt_FasterRCNN_COCO_no_filter.csv')
                original_video = KittiVideo(
                    video_name, ORIGINAL_RESOL, dt_file, img_path,
                    filter_flag=True, merge_label_flag=True)
                if original_video.duration < 10:
                    continue
                videos = {}
                for resol in RESOLUTION_LIST:
                    img_path = os.path.join(
                        seg_path, 'image_02', 'data', resol)
                    dt_file = os.path.join(
                        seg_path, 'image_02', 'data', resol, 'profile',
                        f'updated_gt_FasterRCNN_COCO_no_filter.csv')
                    video = KittiVideo(video_name, resol, dt_file,
                                       img_path, filter_flag=True,
                                       merge_label_flag=True)
                    videos[resol] = video
                    print('loading {}...'.format(dt_file))
                start_frame = original_video.start_frame_index
                end_frame = original_video.end_frame_index
                print('{} start={} end={}'.format(
                    video_name, start_frame, end_frame))
                # use 10 seconds video for profiling
                profile_start = start_frame
                profile_end = start_frame + \
                    original_video.frame_rate * \
                    int(original_video.duration/3) - 1

                print('profile {} start={} end={}'.format(
                    video_name, profile_start, profile_end))
                best_resol, best_fps, best_bw = pipeline.profile(
                    video_name, videos, original_video, [profile_start,
                                                         profile_end])

                print("Profile {}: best resol={}, best fps={}, best bw={}"
                      .format(video_name, best_resol, best_fps, best_bw))

                test_start = profile_end + 1
                test_end = end_frame

                print('Evaluate {} start={} end={}'.format(
                    video_name, test_start, test_end))
                f1_score, relative_bw = pipeline.evaluate(
                    os.path.join(OUTPUT_PATH, video_name +
                                 '.mp4'), original_video,
                    videos[str(best_resol[1]) + 'p'], best_fps,
                    [test_start, test_end])

                print('{} best fps={}, best resolution={} ==> tested f1={}'
                      .format(video_name, best_fps/original_video.frame_rate,
                              best_resol, f1_score))
                writer.writerow([video_name, str(best_resol[1]) + 'p',
                                 f1_score,
                                 best_fps / original_video.frame_rate,
                                 relative_bw])


if __name__ == '__main__':
    main()
# save for future use
# def parse_args():
#     """Parse args."""
#     parser = argparse.ArgumentParser(
#         description="Awstream with spatial overfitting")
#     parser.add_argument("--video", type=str, required=True, help="video name")
#     parser.add_argument("--output", type=str, required=True,
#                         help="output file")
#     parser.add_argument("--log", type=str, required=True, help="log file")
#     parser.add_argument("--short_video_length", type=int, required=True,
#                         help="short video length in second")
#     parser.add_argument("--profile_length", type=int, required=True,
#                         help="profile length in second")
#     args = parser.parse_args()
#     return args
