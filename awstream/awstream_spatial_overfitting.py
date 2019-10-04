'''
This script is to used to compute overfitting result of each video. They are
used to draw features vs awstream bandwidth
'''

from awstream.profiler import VideoConfig, profile, profile_eval, \
    select_best_config
from collections import defaultdict
from utils.model_utils import load_full_model_detection
# load_full_model_detection_new
from utils.utils import load_metadata
# from matplotlib import cm
# import matplotlib.pyplot as plt
# import numpy as np
# import os
import pdb
# import subprocess
# import sys
import argparse


DRVING_VIDEOS = ['driving_downtown', 'nyc', 'driving1', 'driving2',
                 'park', 'lane_split']

PATH = '/mnt/data/zhujun/dataset/Youtube/'
TEMPORAL_SAMPLING_LIST = [1]
DATASET_LIST = sorted(['traffic', 'jp_hw', 'russia', 'tw_road', 'highway',
                       'tw_under_bridge', 'highway_normal_traffic', 'nyc',
                       'lane_split', 'tw', 'tw1', 'jp', 'russia1', 'park',
                       'driving_downtown', 'drift', 'crossroad4', 'driving1',
                       'crossroad3', 'crossroad2', 'crossroad', 'driving2',
                       'motorway'])
DATASET_LIST = ['highway']
IMAGE_RESOLUTION_DICT = {'360p': [640, 360],
                         '480p': [854, 480],
                         '540p': [960, 540],
                         '576p': [1024, 576],
                         '720p': [1280, 720],
                         '1080p': [1920, 1080],
                         '3840p': [3840, 2160], }

SHORT_VIDEO_LENGTH = 30  # seconds
IOU_THRESH = 0.5
TARGET_F1 = 0.9
PROFILE_LENGTH = 30  # seconds
OFFSET = 0
RESOLUTION_LIST = ['720p', '540p', '480p', '360p']  # '2160p', '1080p',

ORIGINAL_REOSL = '720p'


def resol_str_to_int(resol_str):
    return int(resol_str.strip('p'))


def main():
    parser = argparse.ArgumentParser(description="Awstream with spatial overfitting")
    parser.add_argument("--video", type=str,
                        help="video name")
    # parser.add_argument("--output", type=str,
    #                     help="output result file")
    # parser.add_argument("--start_frame", type=int,
    #                     help="short video length in seconds")
    # parser.add_argument("--end_frame", type=int,
    #                     help="profile length in seconds")
    # parser.add_argument("--resolution", type=str,
    #                      help="video resolution")
    args = parser.parse_args()
    dataset = args.video
    # start_frame = args.start_frame
    # end_frame = args.end_frame
    # with open('model_dependence_results/awstream_spatial_results.csv', 'w', 1) as f:
    # for dataset in DATASET_LIST:
    with open('spatial_overfitting_results_09_28/awstream_spatial_overfitting_{}.csv'.format(dataset), 'w', 1) as f:
        f.write('dataset,best_resolution,f1,frame_rate,bandwidth\n')
        f_profile = open('spatial_overfitting_profile_09_28/awstream_spatial_overfitting_profile_{}.csv'
                         .format(dataset), 'w', 1)
        f_profile.write('dataset,resolution,sample_rate,f1,tp,fp,fn\n')
        metadata = load_metadata(PATH + dataset + '/metadata.json')
        resolution = metadata['resolution']
        height = metadata['resolution'][1]
        original_resol = ORIGINAL_REOSL
        # load detection results of fasterRCNN + full resolution +
        # highest frame rate as ground truth
        frame_rate = metadata['frame rate']
        frame_cnt = metadata['frame count']
        num_of_short_videos = frame_cnt//(SHORT_VIDEO_LENGTH*frame_rate)

        gt_dict = defaultdict(None)
        dt_dict = defaultdict(None)

        for resol in RESOLUTION_LIST:
            if resol_str_to_int(resol) > height:
                continue
            dt_file = PATH + dataset + '/' + resol + \
                '/profile/updated_gt_FasterRCNN_COCO.csv'
            gt_file = PATH + dataset + '/' + original_resol + \
                '/profile/updated_gt_FasterRCNN_COCO.csv'
            if dataset in DRVING_VIDEOS:
                gt_dict[resol], frame_cnt = load_full_model_detection(gt_file)
                dt_dict[resol], frame_cnt = load_full_model_detection(dt_file)
            else:
                gt_dict[resol], frame_cnt = load_full_model_detection(gt_file, height=resolution[1])
                dt_dict[resol], frame_cnt = load_full_model_detection(dt_file, height=resol_str_to_int(resol))

        print('Processing', dataset)
        for i in range(num_of_short_videos):
            clip = dataset + '_' + str(i)
            start_frame = i * SHORT_VIDEO_LENGTH * frame_rate + 1 + \
                OFFSET * frame_rate
            end_frame = (i+1) * SHORT_VIDEO_LENGTH * frame_rate + \
                OFFSET * frame_rate
            print('short video start={} end={}'
                  .format(start_frame, end_frame))
            # use 30 seconds video for profiling
            profile_start = start_frame
            profile_end = start_frame + frame_rate * PROFILE_LENGTH - 1

            original_config = VideoConfig(IMAGE_RESOLUTION_DICT[original_resol], frame_rate)
            gt = gt_dict[str(original_config.resolution[1]) + 'p']

            configs = profile(clip, gt, dt_dict, original_config,
                              profile_start, profile_end, f_profile,
                              RESOLUTION_LIST, TEMPORAL_SAMPLING_LIST)

            for c in configs:
                c.debug_print()
            img_path_dict = defaultdict(None)
            for c in configs:
                if c.resolution[1] > height:
                    continue
                resol = str(c.resolution[1]) + 'p'
                img_path_dict[resol] = PATH + dataset + '/' + resol + '/'

            best_config, best_bw = select_best_config(clip, img_path_dict,
                                                      original_config,
                                                      configs,
                                                      profile_start,
                                                      profile_end)

            print("Finished profiling on frame [{},{}]."
                  .format(profile_start, profile_end))
            print("best resol={}, best fps={}, best bw={}"
                  .format(best_config.resolution, best_config.fps,
                          best_bw))

            test_start = profile_start
            test_end = profile_end

            dt = dt_dict[str(best_config.resolution[1]) + 'p']
            original_config.debug_print()
            best_config.debug_print()

            f1, relative_bw = profile_eval(clip, img_path_dict, gt, dt,
                                           original_config, best_config,
                                           test_start, test_end)

            print("Finished testing {} frame [{},{}], f1={}, bw={}."
                  .format(dataset, test_start, test_end, f1, relative_bw))

            print('{} best fps={}, best resolution={} ==> tested f1={}'
                  .format(dataset+str(i), best_config.fps/frame_rate,
                          best_config.resolution, f1))
            f.write(clip + ',' + str(best_config.resolution[1]) + 'p' + ','
                    + str(f1) + ',' + str(best_config.fps) + ',' +
                    str(relative_bw) + '\n')


if __name__ == '__main__':
    main()
