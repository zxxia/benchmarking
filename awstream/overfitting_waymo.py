'''
This script is to used to compute overfitting result of waymo dataset. They are
used to draw features vs awstream bandwidth
'''

from awstream.profiler import VideoConfig, profile, profile_eval, \
    select_best_config
from collections import defaultdict
from utils.model_utils import load_full_model_detection, load_waymo_detection
# load_full_model_detection_new
from utils.utils import load_metadata
# from matplotlib import cm
# import matplotlib.pyplot as plt
# import numpy as np
# import os
import pdb
# import subprocess
# import sys
import glob
import re

PATH = '/mnt/data/zhujun/new_video/'
TEMPORAL_SAMPLING_LIST = [1]  # [20,15,10,5,4,3,2.5,2,1.8,1.5,1.2,1]
DATASET_LIST = ['training_0000', 'training_0001',
                'training_0002', 'validation_0000']

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
OFFSET = 0  # 1*60+30
RESOLUTION_LIST = ['720p', '540p', '480p', '360p']  # '2160p', '1080p',
WAYMO_FRONT_CAMERA_FRAME_RATE = 10
WAYMO_FRONT_CAMERA_RESOLUTION = [1920, 1280]


def resol_str_to_int(resol_str):
    return int(resol_str.strip('p'))


def main():
    with open('waymo_overfitting/waymo_overfitting_spatial.csv', 'w') as f:
        f.write('dataset,best_resolution,f1,frame_rate,bandwidth\n')
        for dataset in DATASET_LIST:
            f_profile = open('waymo_overfitting/awstream_profile_spatial_{}.csv'
                             .format(dataset), 'w')

            segment_files = sorted(glob.glob(PATH + dataset + '/*.tfrecord'))
            for segment_file in segment_files:
                rz = re.findall("(segment-.*)_with_camera_labels.tfrecord", segment_file)
                segment_name = rz[0]
                print(rz[0])
                resolution = WAYMO_FRONT_CAMERA_RESOLUTION
                height = resolution[1]
                original_resol = '720p'  # str(height) + 'p'
                # load detection results of fasterRCNN + full resolution +
                # highest frame rate as ground truth
                frame_rate = WAYMO_FRONT_CAMERA_FRAME_RATE
                # frame_cnt = metadata['frame count']
                num_of_short_videos = 1  # frame_cnt//(SHORT_VIDEO_LENGTH*frame_rate)

                gt_dict = defaultdict(None)
                dt_dict = defaultdict(None)

                for resol in RESOLUTION_LIST:
                    if resol_str_to_int(resol) > height:
                        continue
                    dt_file = PATH + dataset + '/' + segment_name + '/' + resol + \
                        '/profile/updated_gt_FasterRCNN_COCO.csv'
                    gt_file = PATH + dataset + '/' + segment_name + '/' + original_resol + \
                        '/profile/updated_gt_FasterRCNN_COCO.csv'
                    # gt_file = PATH + 'waymo_groundtruths/' + dataset + '/' + original_resol + \
                        # '/gt_{}.csv'.format(segment_name)
                    # gt_dict[resol], frame_cnt = load_waymo_detection(gt_file)
                    gt_dict[resol], frame_cnt = load_full_model_detection(gt_file)
                    dt_dict[resol], frame_cnt = load_full_model_detection(dt_file)

                print('Processing', dataset)
                test_bw_list = list()
                test_f1_list = list()
                for i in range(num_of_short_videos):
                    # start_frame = i * SHORT_VIDEO_LENGTH * frame_rate + 1 + \
                        # OFFSET * frame_rate
                    # end_frame = (i+1) * SHORT_VIDEO_LENGTH * frame_rate + \
                        # OFFSET * frame_rate
                    start_frame = 1
                    end_frame = frame_cnt
                    print('short video start={} end={}'
                          .format(start_frame, end_frame))
                    # use 30 seconds video for profiling
                    # pdb.set_trace()
                    profile_start = start_frame
                    profile_end = end_frame  # start_frame + frame_rate * PROFILE_LENGTH - 1

                    original_config = VideoConfig(IMAGE_RESOLUTION_DICT[original_resol], frame_rate)

                    configs = profile(gt_dict, dt_dict, original_config,
                                      profile_start, profile_end, f_profile,
                                      RESOLUTION_LIST, TEMPORAL_SAMPLING_LIST)
                    if not configs:
                        continue
                    for c in configs:
                        c.debug_print()
                    img_path_dict = defaultdict(None)
                    for c in configs:
                        if c.resolution[1] > height:
                            continue
                        resol = str(c.resolution[1]) + 'p'
                        img_path_dict[resol] = PATH + dataset + '/' + segment_name + '/' + resol + '/'


                    best_config, best_bw = select_best_config(img_path_dict,
                                                              original_config,
                                                              configs,
                                                              profile_start,
                                                              profile_end)

                    print("Finished profiling on frame [{},{}]."
                          .format(profile_start, profile_end))
                    print("best resol={}, best fps={}, best bw={}"
                          .format(best_config.resolution, best_config.fps,
                                  best_bw))

                    # test on the whole video
                    test_start = profile_end + 1
                    test_end = end_frame

                    gt = gt_dict[str(original_config.resolution[1]) + 'p']
                    dt = dt_dict[str(best_config.resolution[1]) + 'p']
                    # pdb.set_trace()

                    f1, relative_bw = profile_eval(img_path_dict, gt, dt,
                                                   original_config, best_config,
                                                   profile_start, profile_end)

                    print("Finished testing on frame [{},{}]."
                          .format(test_start, test_end))
                    test_bw_list.append(relative_bw)
                    test_f1_list.append(f1)

                    print('{} best fps={}, best resolution={} ==> tested f1={}'
                          .format(dataset+str(i), best_config.fps/frame_rate,
                                  best_config.resolution, f1))
                    f.write(dataset + '_' + str(i) + ',' +
                            str(best_config.resolution[1]) + 'p' + ',' + str(f1) +
                            ',' + str(best_config.fps) + ',' +
                            str(relative_bw) + '\n')


if __name__ == '__main__':
    main()
