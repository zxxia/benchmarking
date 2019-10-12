'''
This script is to used to compute overfitting result of each video.
They are used to draw features vs awstream bandwidth
'''

import argparse
from collections import defaultdict
import pdb
from awstream.profiler import VideoConfig, profile, profile_eval, \
    select_best_config
from constants import RESOL_DICT
from utils.model_utils import load_full_model_detection, \
    filter_video_detections
# load_full_model_detection_new
from utils.utils import load_metadata
# from matplotlib import cm
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import subprocess
# import sys


DRVING_VIDEOS = ['driving_downtown', 'nyc', 'driving1', 'driving2',
                 'park', 'lane_split']

# PATH = '/mnt/data/zhujun/dataset/Youtube/'
PATH = '/data/zxxia/benchmarking/results/videos/'
DATA_PATH = '/data/zxxia/videos/'
TEMPORAL_SAMPLING_LIST = [1]
DATASET_LIST = sorted(['traffic', 'jp_hw', 'russia', 'tw_road', 'highway',
                       'tw_under_bridge', 'highway_normal_traffic', 'nyc',
                       'lane_split', 'tw', 'tw1', 'jp', 'russia1', 'park',
                       'driving_downtown', 'drift', 'crossroad4', 'driving1',
                       'crossroad3', 'crossroad2', 'crossroad', 'driving2',
                       'motorway'])
DATASET_LIST = ['highway']

SHORT_VIDEO_LENGTH = 30  # seconds
IOU_THRESH = 0.5
TARGET_F1 = 0.9
PROFILE_LENGTH = 30  # seconds
OFFSET = 0
# RESOLUTION_LIST = ['720p', '540p', '480p', '360p']  # '2160p', '1080p',
RESOLUTION_LIST = ['720p', '300p']  # '2160p', '1080p',

ORIGINAL_REOSL = '720p'


def resol_str_to_int(resol_str):
    """ resolution string to integer """
    return int(resol_str.strip('p'))


def main():
    parser = argparse.ArgumentParser(
        description="Awstream with spatial overfitting")
    parser.add_argument("--video", type=str, help="video name")
    parser.add_argument("--output", type=str, help="output result file")
    parser.add_argument("--log", type=str, help="log file")
    args = parser.parse_args()
    dataset = args.video
    output_file = args.output
    log_file = args.log
    with open(output_file, 'w', 1) as f_out:
        f_out.write('dataset,best_resolution,f1,frame_rate,bandwidth\n')
        f_profile = open(log_file, 'w', 1)
        f_profile.write('dataset,resolution,sample_rate,f1,tp,fp,fn\n')
        metadata = load_metadata(DATA_PATH + dataset + '/metadata.json')
        resolution = metadata['resolution']
        height = resolution[1]
        original_resol = ORIGINAL_REOSL
        # load detection results of fasterRCNN + full resolution +
        # highest frame rate as ground truth
        fps = metadata['frame rate']
        frame_cnt = metadata['frame count']
        num_of_short_videos = frame_cnt//(SHORT_VIDEO_LENGTH*fps)
        gt_file = PATH + dataset + '/' + original_resol + \
            '/profile/updated_gt_FasterRCNN_COCO_no_filter.csv'
        print(gt_file)
        gt, frame_cnt = load_full_model_detection(gt_file)
        gt = filter_video_detections(gt, target_types={3, 8},
                                     height_range=(720//20, 720))
        for frame_idx, boxes in gt.items():
            for box_idx, box in enumerate(boxes):
                gt[frame_idx][box_idx][4] = 3

        dt_dict = defaultdict(None)
        for resol in RESOLUTION_LIST:
            cur_h = resol_str_to_int(resol)
            print(cur_h)
            if cur_h > height:
                continue
            dt_file = PATH + dataset + '/' + resol + \
                '/profile/updated_gt_FasterRCNN_COCO_no_filter.csv'
            print(dt_file)
            if dataset in DRVING_VIDEOS:
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
                for box_idx, box in enumerate(boxes):
                    dt_dict[resol][frame_idx][box_idx][4] = 3

        print('Processing', dataset)
        for i in range(num_of_short_videos):
            clip = dataset + '_' + str(i)
            start_frame = i * SHORT_VIDEO_LENGTH * fps + 1 + \
                OFFSET * fps
            end_frame = (i+1) * SHORT_VIDEO_LENGTH * fps + \
                OFFSET * fps
            print('short video start={} end={}'
                  .format(start_frame, end_frame))
            # use 30 seconds video for profiling
            profile_start = start_frame
            profile_end = start_frame + fps * PROFILE_LENGTH - 1

            original_config = VideoConfig(RESOL_DICT[original_resol], fps)

            configs = profile(clip, gt, dt_dict, original_config,
                              profile_start, profile_end, f_profile,
                              RESOLUTION_LIST, TEMPORAL_SAMPLING_LIST)

            img_path_dict = defaultdict(None)
            for c in configs:
                if c.resolution[1] > height:
                    continue
                resol = str(c.resolution[1]) + 'p'
                img_path_dict[resol] = DATA_PATH + dataset + '/' + resol + '/'
            img_path_dict[original_resol] = DATA_PATH + dataset + '/' + original_resol + '/'
            # img_path_dict['300p'] = DATA_PATH + dataset + '/' + '300p' + '/'

            best_config, best_bw = select_best_config(clip, img_path_dict,
                                                      original_config, configs,
                                                      profile_start,
                                                      profile_end)

            print("Finished profiling on frame [{},{}]."
                  .format(profile_start, profile_end))
            print("best resol={}, best fps={}, best bw={}"
                  .format(best_config.resolution, best_config.fps,
                          best_bw))

            test_start = profile_start
            test_end = profile_end

            # best_config = VideoConfig(RESOL_DICT['300p'], fps)
            dets = dt_dict[str(best_config.resolution[1]) + 'p']
            original_config.debug_print()
            best_config.debug_print()

            f1_score, relative_bw = profile_eval(clip, img_path_dict, gt, dets,
                                                 original_config, best_config,
                                                 test_start, test_end)

            print("Finished testing {} frame [{},{}], f1={}, bw={}."
                  .format(dataset, test_start, test_end,
                          f1_score, relative_bw))

            print('{} best fps={}, best resolution={} ==> tested f1={}'
                  .format(dataset+str(i), best_config.fps/fps,
                          best_config.resolution, f1_score))
            f_out.write(','.join([clip, str(best_config.resolution[1]) + 'p',
                                  str(f1_score), str(best_config.fps),
                                  str(relative_bw)]) + '\n')


if __name__ == '__main__':
    main()
