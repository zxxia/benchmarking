"""NoScope Motivation Script. """

import argparse
import csv
import pdb
import sys
import os
import copy
import numpy as np
import glob
from collections import defaultdict
from benchmarking.noscope.Noscope import NoScope
from benchmarking.video import KittiVideo

THRESH_LIST = np.arange(0.3, 1.1, 0.1)
MSE_list = [0, 25, 50, 75, 100, 125, 150]
OFFSET = 0  # The time offset from the start of the video. Unit: seconds
ORIGINAL_RESOL = '720p'
DT_ROOT = '/mnt/data/zhujun/dataset/KITTI'
SMALL_MODEL_PATH = '/mnt/data/zhujun/dataset/NoScope_finetuned_models'
PROFILE_VIDEO_SAVEPATH = '/mnt/data/zhujun/dataset/NoScope_finetuned_models/original_profile_videos/'

def main():
    """NoScope."""
    name = 'KITTI'
    f_out = open('./results/Noscope_e2e_result_KITTI.csv', 'w')
    f_out.write('dataset,best_frame_diff,best_confidence_score_thresh,f1,bandwidth,gpu,selected_frames,triggered_frames\n')
    OUTPUT_PATH = os.path.join(
        SMALL_MODEL_PATH, name, 'data'
    )
    LOCATIONS = ['City', 'Residential', 'Road']
    for loc in LOCATIONS:
        for seg_path in sorted(glob.glob(os.path.join(DT_ROOT, loc, '*'))):
            if not os.path.isdir(seg_path):
                continue
            video_name = loc + '_' + os.path.basename(seg_path).split('_')[-2]
            pipeline = NoScope(THRESH_LIST, MSE_list, OUTPUT_PATH + '/tmp_log_' + video_name + '.csv')
            print('processing ', video_name)
            img_path = os.path.join(
                    seg_path, 'image_02', 'data', ORIGINAL_RESOL)
            dt_file = os.path.join(
                    seg_path, 'image_02', 'data', ORIGINAL_RESOL, 'profile',
                    'updated_gt_FasterRCNN_COCO_no_filter.csv') 
            original_video = KittiVideo(
                    video_name, ORIGINAL_RESOL, dt_file, img_path)


            dt_file = os.path.join(
                    SMALL_MODEL_PATH, 'KITTI',  'data', video_name,
                    'updated_gt_mobilenetFinetuned_COCO_no_filter.csv') 
            new_mobilenet_video = KittiVideo(video_name, ORIGINAL_RESOL, dt_file, img_path)                           
            if original_video.duration < 10:
                    continue

            start_frame = original_video.start_frame_index
            end_frame = original_video.end_frame_index
            profile_start = start_frame
            profile_end = start_frame + \
                original_video.frame_rate * \
                int(original_video.duration/3) - 1
            print('profile {} start={} end={}'.format(
                    video_name, profile_start, profile_end))


            best_mse_thresh, best_thresh, best_f1, best_relative_bw, best_gpu = \
                pipeline.profile(video_name, original_video, new_mobilenet_video,
                                 [profile_start, profile_end],
                                 profile_video_savepath=PROFILE_VIDEO_SAVEPATH)

            print("Profile {}: best mse thresh={}, best thresh={}, best bw={}"
                  .format(video_name, best_mse_thresh, best_thresh, best_relative_bw))

            test_start = profile_end + 1
            test_end = end_frame

            print('Evaluate {} start={} end={}'.format(
                video_name, test_start, test_end))
            f1_score, relative_bw, relative_gpu, selected_frame_list, trigger_frame_list = pipeline.evaluate(
                os.path.join(OUTPUT_PATH, video_name + '.mp4'), original_video,
                new_mobilenet_video, best_mse_thresh, best_thresh,
                [test_start, test_end])

            print('{} best thresh={} ==> tested f1={}'
                  .format(video_name, best_thresh, f1_score))
            f_out.write(','.join([video_name,
                                  str(best_mse_thresh), 
                                  str(best_thresh),
                                  str(f1_score),
                                  str(relative_bw),
                                  str(relative_gpu),
                                  ' '.join([str(x) for x in selected_frame_list]),
                                  ' '.join([str(x) for x in trigger_frame_list])])+ '\n')

              


if __name__ == '__main__':
    main()
