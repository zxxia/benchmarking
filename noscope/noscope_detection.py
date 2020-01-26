"""NoScope Overfitting Script. """

import argparse
import csv
import pdb
import sys
import os
import copy
import numpy as np
from collections import defaultdict
from benchmarking.noscope.Noscope import NoScope
from benchmarking.video import YoutubeVideo

THRESH_LIST = np.arange(0.3, 1.1, 0.1)
MSE_list = [0]
OFFSET = 0  # The time offset from the start of the video. Unit: seconds
# VIDEOS = ['crossroad', 'crossroad2', 'crossroad3', 'crossroad4', 'drift',
#           'driving1', 'driving_downtown', 'highway',
#           'nyc', 'jp',  'lane_split',  'driving2',
#           'motorway', 'park', 'russia', 'russia1', 
#           'traffic', 'tw', 'tw1',
#           'tw_under_bridge']

VIDEOS = ['crossroad', 'crossroad2', 'crossroad2_night', 'crossroad3', 'crossroad4',
        'drift', 'driving1', 'driving2', 'driving_downtown', 'jp', 
        'nyc', 'park', 'tw1']
DT_ROOT = '/mnt/data/zhujun/dataset/Youtube'
SHORT_VIDEO_LENGTH = 30
profile_length = 30
SMALL_MODEL_PATH = '/mnt/data/zhujun/dataset/NoScope_finetuned_models'
PROFILE_VIDEO_SAVEPATH = '/mnt/data/zhujun/dataset/NoScope_finetuned_models/profile_videos_30s/'
def main():
    """NoScope."""
    f_out = open('./results/Noscope_overfitting_result_allvideo_w_gpu_cost_min_gpu.csv', 'w')
    f_out.write('dataset,best_frame_diff,best_confidence_score_thresh,f1,bandwidth,gpu,selected_frames,triggered_frames\n')
    for name in VIDEOS:
        if "cropped" in name:
            resol = '360p'
        else:
            resol = '720p'
        OUTPUT_PATH = os.path.join(
            SMALL_MODEL_PATH, name, 'data'
        )
        pipeline = NoScope(THRESH_LIST, MSE_list, OUTPUT_PATH + '/tmp_log_overfitting_w_gpu_cost_min_gpu.csv')
        metadata_file = DT_ROOT + '/{}/metadata.json'.format(name)
        img_path = os.path.join(DT_ROOT, name, resol)
        dt_file = os.path.join(
            DT_ROOT, name, resol,
            'profile/updated_gt_FasterRCNN_COCO_no_filter.csv')
        original_video = YoutubeVideo(name, resol, metadata_file, dt_file, img_path)


        if name == 'crossroad2':
            dt_file = os.path.join(
                SMALL_MODEL_PATH, name, 
                'data/updated_gt_mobilenetFinetuned_by_nighttimedata_COCO_no_filter.csv')   
        elif name == 'crossroad2_night':
            dt_file = os.path.join(
                SMALL_MODEL_PATH, name, 
                'data/updated_gt_mobilenetFinetuned_by_daytimedata_COCO_no_filter.csv')
        else:
            dt_file = os.path.join(
                SMALL_MODEL_PATH, name, 
                'data/updated_gt_mobilenetFinetuned_COCO_no_filter.csv')
        

        new_mobilenet_video = YoutubeVideo(name, resol, metadata_file, dt_file, img_path)

        num_of_short_videos = original_video.frame_count // (
            SHORT_VIDEO_LENGTH*original_video.frame_rate)

        for i in range(num_of_short_videos):
            clip = name + '_' + str(i)
            start_frame = i*SHORT_VIDEO_LENGTH * \
                original_video.frame_rate+1+OFFSET*original_video.frame_rate
            end_frame = (i+1)*SHORT_VIDEO_LENGTH * \
                original_video.frame_rate+OFFSET*original_video.frame_rate
            print('{} start={} end={}'.format(clip, start_frame, end_frame))

            # use profile_length video for profiling
            profile_start = start_frame
            profile_end = end_frame

            print('profile {} start={} end={}'.format(
                clip, profile_start, profile_end))
            best_mse_thresh, best_thresh, best_f1, best_relative_bw, best_gpu = \
                pipeline.profile(clip, original_video, new_mobilenet_video,
                                 [profile_start, profile_end],
                                 profile_video_savepath=PROFILE_VIDEO_SAVEPATH)

            print("Profile {}: f1={}, bw={}, gpu={}"
                  .format(clip, best_f1, best_relative_bw, best_gpu))

            f_out.write(','.join([clip,
                                  str(best_mse_thresh), 
                                  str(best_thresh),
                                  str(best_f1),
                                  str(best_relative_bw),
                                  str(best_gpu)])+ '\n')

              


if __name__ == '__main__':
    main()
