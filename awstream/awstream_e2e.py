''' This script is to used to compute spatial overfitting result of each video.
They are used to draw features vs awstream bandwidth
'''
# from collections import defaultdict
import argparse
import os
# import pdb
# from utils.model_utils import filter_video_detections, remove_overlappings
from benchmarking.awstream.Awstream import Awstream
from benchmarking.video import YoutubeVideo


# DATA_PATH = '/mnt/data/zhujun/dataset/Youtube/'
PATH = '/data/zxxia/benchmarking/results/videos'
# PATH = '/home/zxxia/benchmarking/videos/'
DATA_PATH = '/data2/zxxia/videos'
# TEMPORAL_SAMPLING_LIST = [1]
TEMPORAL_SAMPLING_LIST = [20, 15, 10, 5, 4, 3, 2.5, 2, 1.8, 1.5, 1.2, 1]
OFFSET = 0
RESOLUTION_LIST = ['720p', '540p', '480p', '360p']  # '2160p', '1080p',
# RESOLUTION_LIST = ['720p']  # '2160p', '1080p',
ORIGINAL_REOSL = '720p'
OUTPUT_PATH = '/data/zxxia/benchmarking/awstream/short_videos'


def parse_args():
    """Parse args."""
    parser = argparse.ArgumentParser(
        description="Awstream with spatial overfitting")
    parser.add_argument("--video", type=str, required=True, help="video name")
    parser.add_argument("--output", type=str, required=True,
                        help="output file")
    parser.add_argument("--log", type=str, required=True, help="log file")
    parser.add_argument("--short_video_length", type=int, required=True,
                        help="short video length in second")
    parser.add_argument("--profile_length", type=int, required=True,
                        help="profile length in second")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    pipeline = Awstream(TEMPORAL_SAMPLING_LIST,
                        RESOLUTION_LIST, None, args.log)
    with open(args.output, 'w', 1) as f_out:
        f_out.write('dataset,best_resolution,f1,frame_rate,bandwidth\n')
        metadata_file = os.path.join(DATA_PATH, args.video, 'metadata.json')
        # loading videos
        img_path = os.path.join(DATA_PATH, args.video, ORIGINAL_REOSL)
        dt_file = os.path.join(PATH, args.video, ORIGINAL_REOSL, 'profile',
                               'updated_gt_FasterRCNN_COCO_no_filter.csv')
        original_video = YoutubeVideo(args.video, ORIGINAL_REOSL,
                                      metadata_file, dt_file, img_path,
                                      filter_flag=True,
                                      merge_label_flag=True)
        videos = {}
        for resol in RESOLUTION_LIST:
            img_path = os.path.join(DATA_PATH, args.video, resol)
            dt_file = os.path.join(PATH, args.video, resol, 'profile',
                                   'updated_gt_FasterRCNN_COCO_no_filter.csv')
            video = YoutubeVideo(args.video, resol, metadata_file, dt_file,
                                 img_path, filter_flag=True,
                                 merge_label_flag=True)
            videos[resol] = video
            print('loading {}...'.format(dt_file))

        num_of_short_videos = original_video.frame_count // (
            args.short_video_length*original_video.frame_rate)

        for i in range(num_of_short_videos):
            clip = args.video + '_' + str(i)
            start_frame = i*args.short_video_length * \
                original_video.frame_rate+original_video.start_frame_index + \
                OFFSET*original_video.frame_rate
            end_frame = (i+1)*args.short_video_length * \
                original_video.frame_rate+OFFSET*original_video.frame_rate
            print('{} start={} end={}'.format(clip, start_frame, end_frame))
            # use 30 seconds video for profiling
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

            test_start = profile_end + 1
            test_end = end_frame

            print('Evaluate {} start={} end={}'.format(
                clip, test_start, test_end))
            f1_score, relative_bw = pipeline.evaluate(
                os.path.join(OUTPUT_PATH, clip + '.mp4'), original_video,
                videos[str(best_resol[1])+'p'], best_fps,
                [test_start, test_end])

            print('{} best fps={}, best resolution={} ==> tested f1={}'
                  .format(clip, best_fps/original_video.frame_rate,
                          best_resol, f1_score))
            f_out.write(','.join([clip, str(best_resol[1]) + 'p',
                                  str(f1_score),
                                  str(best_fps / original_video.frame_rate),
                                  str(relative_bw)]) + '\n')


if __name__ == '__main__':
    main()
