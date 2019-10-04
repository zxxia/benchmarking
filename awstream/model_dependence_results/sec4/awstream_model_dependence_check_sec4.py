'''
This script is used to test model dependency in paper section 4
'''

from awstream.profiler import VideoConfig, profile, profile_eval, \
    select_best_config
from collections import defaultdict
from utils.model_utils import load_full_model_detection
# load_full_model_detection_new
from utils.utils import load_metadata
# from matplotlib import cm
# import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
# import subprocess
# import sys
import cv2

PATH = '/mnt/data/zhujun/dataset/Youtube/'
TEMPORAL_SAMPLING_LIST = [20, 15, 10, 5, 4, 3, 2.5, 2, 1.8, 1.5, 1.2, 1]
DATASET_LIST = sorted(['traffic', 'jp_hw', 'russia', 'tw_road', 'highway',
                       'tw_under_bridge', 'highway_normal_traffic', 'nyc',
                       'lane_split', 'tw', 'tw1', 'jp', 'russia1', 'park',
                       'driving_downtown', 'drift', 'crossroad4', 'driving1',
                       'crossroad3', 'crossroad2', 'crossroad', 'driving2',
                       'motorway'])
DATASET_LIST = ['motorway', 'park', 'highway', 'nyc']
# DATASET_LIST = ['highway']
DATASET_LIST = ['motorway']
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
# TARGET_F1 = 0.8
PROFILE_LENGTH = 30  # seconds
OFFSET = 0  # 1*60+30
RESOLUTION_LIST = ['720p', '540p', '480p', '360p']  # '2160p', '1080p',

ORIGINAL_REOSL = '720p'
SELECTED_REOSL = '360p'
divisor = 3


def resol_str_to_int(resol_str):
    return int(resol_str.strip('p'))


def main():
    # with open('model_dependence_results/awstream_spatial_results.csv', 'w', 1) as f:
    with open('model_dependence_data_sample_rate_{}.csv'.format(divisor), 'w', 1) as f:
        f.write('dataset,temporal no spatial bw,temporal no spatial f1,'
                'temporal spatial bw,temporal spatial f1,temporal cropped bw, temporal cropped f1\n')
        for dataset in DATASET_LIST:
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
                gt_file = PATH + dataset + '/' + resol + \
                    '/profile/updated_gt_FasterRCNN_COCO.csv'
                gt_dict[resol] = load_full_model_detection(gt_file)[0]
                dt_dict[resol] = load_full_model_detection(dt_file)[0]

            print('Processing', dataset)
            for i in range(num_of_short_videos):
                # if i >= 20:
                    # break
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
                # test_start = profile_end + 1
                # test_end = end_frame

                img_path_dict = defaultdict(None)
                img_path_dict[original_resol] = PATH + dataset + '/' + original_resol + '/'
                img_path_dict[SELECTED_REOSL] = PATH + dataset + '/' + SELECTED_REOSL + '/'

                original_config = VideoConfig(IMAGE_RESOLUTION_DICT[original_resol], frame_rate)
                temporal_no_spatial_config = VideoConfig(IMAGE_RESOLUTION_DICT[original_resol], frame_rate/divisor)
                gt = dt_dict[str(original_config.resolution[1]) + 'p']
                dt = dt_dict[str(temporal_no_spatial_config.resolution[1]) + 'p']

                out = cv2.VideoWriter('tmp1.mp4', cv2.VideoWriter_fourcc(*'mp4v'),
                                      frame_rate, (1280, 720))
                for j in range(start_frame, end_frame+1):
                    img_name = PATH + dataset + '/720p/{:06d}.jpg'.format(j)
                    # print(img_name)
                    img = cv2.imread(img_name)
                    out.write(img)
                out.release()
                video_size = os.path.getsize("tmp1.mp4")
                # pdb.set_trace()

                out = cv2.VideoWriter('tmp2.mp4', cv2.VideoWriter_fourcc(*'mp4v'),
                                      frame_rate, (1280, 720))
                for j in range(start_frame, end_frame+1):
                    if j % divisor == 1:
                        img_name = PATH + dataset + '/720p/{:06d}.jpg'.format(j)
                        # print(img_name)
                        img = cv2.imread(img_name)
                        out.write(img)
                out.release()
                video_size_temporal = os.path.getsize("tmp2.mp4")

                temporal_no_spatial_f1, temporal_no_spatial_relative_bw = profile_eval(img_path_dict, gt, dt,
                                               original_config, temporal_no_spatial_config,
                                               profile_start, profile_end)

                spatial_config = VideoConfig(IMAGE_RESOLUTION_DICT[SELECTED_REOSL], frame_rate)
                temporal_spatial_config = VideoConfig(IMAGE_RESOLUTION_DICT[SELECTED_REOSL], frame_rate/divisor)

                gt = dt_dict[str(spatial_config.resolution[1]) + 'p']
                dt = dt_dict[str(temporal_spatial_config.resolution[1]) + 'p']
                out = cv2.VideoWriter('tmp3.mp4', cv2.VideoWriter_fourcc(*'mp4v'),
                                      frame_rate, (640, 360))
                for j in range(start_frame, end_frame+1):
                    img_name = PATH + dataset + '/360p/{:06d}.jpg'.format(j)
                    # print(img_name)
                    img = cv2.imread(img_name)
                    out.write(img)
                out.release()
                video_size_spatial = os.path.getsize("tmp3.mp4")

                out = cv2.VideoWriter('tmp4.mp4', cv2.VideoWriter_fourcc(*'mp4v'),
                                      frame_rate, (640, 360))
                for j in range(start_frame, end_frame+1):
                    if j % divisor == 1:
                        img_name = PATH + dataset + '/360p/{:06d}.jpg'.format(j)
                        img = cv2.imread(img_name)
                        out.write(img)
                out.release()
                video_size_temporal_spatial = os.path.getsize("tmp4.mp4")


                spatial_f1, spatial_relative_bw = profile_eval(img_path_dict, gt, dt,
                                                               spatial_config, temporal_spatial_config,
                                                               profile_start, profile_end)

                gt = dt_dict[str(original_config.resolution[1]) + 'p']
                dt = dt_dict[str(temporal_no_spatial_config.resolution[1]) + 'p']
                out = cv2.VideoWriter('tmp5.mp4', cv2.VideoWriter_fourcc(*'mp4v'),
                                      frame_rate, (1280, 720))
                for j in range(start_frame, end_frame+1):
                    if j % divisor == 1:
                        img_name = PATH + dataset + '/720p/{:06d}.jpg'.format(j)
                        img = cv2.imread(img_name)
                        mask = np.zeros_like(img, dtype=np.uint8)
                        for box in gt[j]:
                            xmin, ymin, xmax, ymax = box[:4]
                            mask[ymin:ymax, xmin:xmax] = 1
                        img *= mask
                        out.write(img)
                out.release()
                video_size_cropped_temporal = os.path.getsize("tmp5.mp4")

                out = cv2.VideoWriter('tmp6.mp4', cv2.VideoWriter_fourcc(*'mp4v'),
                                      frame_rate, (1280, 720))
                for j in range(start_frame, end_frame+1):
                    img_name = PATH + dataset + '/720p/{:06d}.jpg'.format(j)
                    img = cv2.imread(img_name)
                    mask = np.zeros_like(img, dtype=np.uint8)
                    for box in gt[j]:
                        xmin, ymin, xmax, ymax = box[:4]
                        mask[ymin:ymax, xmin:xmax] = 1
                    img *= mask
                    out.write(img)
                out.release()
                video_size_cropped = os.path.getsize("tmp6.mp4")

                # print(video_size_cropped/video_size)
                # f.write(','.join([clip, str(temporal_no_spatial_relative_bw), str(temporal_no_spatial_f1),
                #                   str(spatial_relative_bw), str(spatial_f1),
                #                   str(video_size_cropped_temporal/video_size_cropped),
                #                   str(temporal_no_spatial_f1)])+'\n')

                f.write(','.join([clip, str(video_size_temporal/video_size), str(temporal_no_spatial_f1),
                                  str(video_size_temporal_spatial/video_size_spatial), str(spatial_f1),
                                  str(video_size_cropped_temporal/video_size_cropped),
                                  str(temporal_no_spatial_f1)])+'\n')
                # pdb.set_trace()


if __name__ == '__main__':
    main()
