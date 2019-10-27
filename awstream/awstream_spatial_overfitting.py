''' This script is to used to compute overfitting result of each video.
They are used to draw features vs awstream bandwidth
'''
from collections import defaultdict
import argparse
import os
# import pdb
from awstream.profiler import VideoConfig, profile, profile_eval, \
    select_best_config
from constants import RESOL_DICT, CAMERA_TYPES
from utils.model_utils import load_full_model_detection, \
    filter_video_detections
from utils.utils import load_metadata, resol_str_to_int


# PATH = '/mnt/data/zhujun/dataset/Youtube/'
PATH = '/data/zxxia/benchmarking/results/videos/'
DATA_PATH = '/data/zxxia/videos/'
TEMPORAL_SAMPLING_LIST = [1]
# DATASET_LIST = sorted(['traffic', 'jp_hw', 'russia', 'tw_road', 'highway',
#                        'tw_under_bridge', 'highway_normal_traffic', 'nyc',
#                        'lane_split', 'tw', 'tw1', 'jp', 'russia1', 'park',
#                        'driving_downtown', 'drift', 'crossroad4', 'driving1',
#                        'crossroad3', 'crossroad2', 'crossroad', 'driving2',
#                        'motorway'])
# DATASET_LIST = ['highway']

# TARGET_F1 = 0.9
OFFSET = 0
RESOLUTION_LIST = ['720p'] #, '540p', '480p', '360p']  # '2160p', '1080p',
# RESOLUTION_LIST = ['720p', '300p']  # '2160p', '1080p',

ORIGINAL_REOSL = '720p'


def parse_args():
    """ parse args """
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
    with open(args.output, 'w', 1) as f_out:
        f_out.write('dataset,best_resolution,f1,frame_rate,bandwidth\n')
        f_profile = open(args.log, 'w', 1)
        f_profile.write('dataset,resolution,sample_rate,f1,tp,fp,fn\n')
        metadata = load_metadata(os.path.join(DATA_PATH, args.video,
                                              'metadata.json'))
        resolution = metadata['resolution']
        original_resol = ORIGINAL_REOSL
        # load detection results of fasterRCNN + full resolution +
        # highest frame rate as ground truth
        fps = metadata['frame rate']
        frame_cnt = metadata['frame count']
        num_of_short_videos = frame_cnt//(args.short_video_length*fps)
        gt_file = os.path.join(PATH, args.video, original_resol, 'profile',
                               'updated_gt_FasterRCNN_COCO_no_filter.csv')
        print(gt_file)
        gtruth, frame_cnt = load_full_model_detection(gt_file)
        gtruth = filter_video_detections(gtruth, target_types={3, 8},
                                         height_range=(720//20, 720))

        # Merge all cars and trucks into cars
        for frame_idx, boxes in gtruth.items():
            for box_idx, _ in enumerate(boxes):
                gtruth[frame_idx][box_idx][4] = 3

        dt_dict = defaultdict(None)
        img_path_dict = defaultdict(None)
        img_path_dict[original_resol] = os.path.join(DATA_PATH, args.video,
                                                     original_resol)
        for resol in RESOLUTION_LIST:
            cur_h = resol_str_to_int(resol)
            # print(cur_h)
            if cur_h > resolution[1]:
                continue
            img_path_dict[resol] = os.path.join(DATA_PATH, args.video, resol)
            dt_file = os.path.join(PATH, args.video, resol, 'profile',
                                   'updated_gt_FasterRCNN_COCO_no_filter.csv')
            print(dt_file)
            if args.video in CAMERA_TYPES['moving']:
                dt_dict[resol], frame_cnt = load_full_model_detection(dt_file)
                dt_dict[resol] = \
                    filter_video_detections(dt_dict[resol],
                                            target_types={3, 8},
                                            height_range=(cur_h//20, cur_h))
            else:
                dt_dict[resol], frame_cnt = load_full_model_detection(dt_file)
                dt_dict[resol] = \
                    filter_video_detections(dt_dict[resol],
                                            target_types={3, 8},
                                            height_range=(cur_h//20, cur_h/2))
            # Merge all cars and trucks into cars
            for frame_idx, boxes in dt_dict[resol].items():
                for box_idx, _ in enumerate(boxes):
                    dt_dict[resol][frame_idx][box_idx][4] = 3

        for i in range(num_of_short_videos):
            clip = args.video + '_' + str(i)
            start_frame = i*args.short_video_length*fps+1+OFFSET*fps
            end_frame = (i+1)*args.short_video_length*fps+OFFSET*fps
            print('{} start={} end={}'.format(clip, start_frame, end_frame))
            # use 30 seconds video for profiling
            profile_start = start_frame
            profile_end = start_frame + fps * args.profile_length - 1

            original_config = VideoConfig(RESOL_DICT[original_resol], fps)

            configs = profile(clip, gtruth, dt_dict, original_config,
                              [profile_start, profile_end], f_profile,
                              RESOLUTION_LIST, TEMPORAL_SAMPLING_LIST)

            best_config, best_bw = select_best_config(clip, img_path_dict,
                                                      original_config, configs,
                                                      [profile_start,
                                                       profile_end])

            print("Profile {}: best resol={}, best fps={}, best bw={}"
                  .format(clip, best_config.resolution, best_config.fps,
                          best_bw))

            test_start = profile_start
            test_end = profile_end

            dets = dt_dict[str(best_config.resolution[1]) + 'p']
            # original_config.debug_print()
            # best_config.debug_print()

            f1_score, relative_bw = profile_eval(clip, img_path_dict, gtruth,
                                                 dets, original_config,
                                                 best_config,
                                                 [test_start, test_end])

            print('{} best fps={}, best resolution={} ==> tested f1={}'
                  .format(clip, best_config.fps/fps,
                          best_config.resolution, f1_score))
            f_out.write(','.join([clip, str(best_config.resolution[1]) + 'p',
                                  str(f1_score), str(best_config.fps),
                                  str(relative_bw)]) + '\n')


if __name__ == '__main__':
    main()
