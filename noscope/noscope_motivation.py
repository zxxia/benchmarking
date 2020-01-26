"""NoScope Motivation Script. """

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
MSE_list = [0, 25, 50, 75, 100, 125, 150]
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
profile_length = 10
SMALL_MODEL_PATH = '/mnt/data/zhujun/dataset/NoScope_finetuned_models'
PROFILE_VIDEO_SAVEPATH = '/mnt/data/zhujun/dataset/NoScope_finetuned_models/original_profile_videos/'
def main():
    """NoScope."""
    f_out = open('./results/Noscope_e2e_result.csv', 'w')
    f_out.write('dataset,best_frame_diff,best_confidence_score_thresh,f1,bandwidth,gpu,selected_frames,triggered_frames\n')
    for name in VIDEOS:
        if "cropped" in name:
            resol = '360p'
        else:
            resol = '720p'
        OUTPUT_PATH = os.path.join(
            SMALL_MODEL_PATH, name, 'data'
        )
        pipeline = NoScope(THRESH_LIST, MSE_list, OUTPUT_PATH + '/tmp_log.csv')
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


        # # profile on first segment 
        # i = 0
        # clip = name + '_' + str(i)
        # start_frame = i*SHORT_VIDEO_LENGTH * \
        #     original_video.frame_rate+1+OFFSET*original_video.frame_rate
        # end_frame = (i+1)*SHORT_VIDEO_LENGTH * \
        #     original_video.frame_rate+OFFSET*original_video.frame_rate
        # print('{} start={} end={}'.format(clip, start_frame, end_frame))

        # # use profile_length video for profiling
        # profile_start = start_frame
        # profile_end = start_frame + original_video.frame_rate * \
        #     profile_length - 1

        # print('profile {} start={} end={}'.format(
        #     clip, profile_start, profile_end))
        # best_mse_thresh, best_thresh, best_relative_bw = \
        #     pipeline.profile(clip, original_video, new_mobilenet_video,
        #                         [profile_start, profile_end],
        #                         profile_video_savepath=PROFILE_VIDEO_SAVEPATH)

        # print("Profile {}: best mse thresh={}, best thresh={}, best bw={}"
        #         .format(clip, best_mse_thresh, best_thresh, best_relative_bw))





        for i in range(num_of_short_videos):
            clip = name + '_' + str(i)
            start_frame = i*SHORT_VIDEO_LENGTH * \
                original_video.frame_rate+1+OFFSET*original_video.frame_rate
            end_frame = (i+1)*SHORT_VIDEO_LENGTH * \
                original_video.frame_rate+OFFSET*original_video.frame_rate
            print('{} start={} end={}'.format(clip, start_frame, end_frame))

            # use profile_length video for profiling
            profile_start = start_frame
            profile_end = start_frame + original_video.frame_rate * \
                profile_length - 1

            print('profile {} start={} end={}'.format(
                clip, profile_start, profile_end))
            best_mse_thresh, best_thresh, best_f1, best_relative_bw, best_gpu = \
                pipeline.profile(clip, original_video, new_mobilenet_video,
                                 [profile_start, profile_end],
                                 profile_video_savepath=PROFILE_VIDEO_SAVEPATH)

            print("Profile {}: best mse thresh={}, best thresh={}, best bw={}"
                  .format(clip, best_mse_thresh, best_thresh, best_relative_bw))

            test_start = profile_end + 1
            test_end = end_frame

            print('Evaluate {} start={} end={}'.format(
                clip, test_start, test_end))
            f1_score, relative_bw, relative_gpu, selected_frame_list, trigger_frame_list = pipeline.evaluate(
                os.path.join(OUTPUT_PATH, clip + '.mp4'), original_video,
                new_mobilenet_video, best_mse_thresh, best_thresh,
                [test_start, test_end])

            print('{} best thresh={} ==> tested f1={}'
                  .format(clip, best_thresh, f1_score))
            f_out.write(','.join([clip,
                                  str(best_mse_thresh), 
                                  str(best_thresh),
                                  str(f1_score),
                                  str(relative_bw),
                                  str(relative_gpu),
                                  ' '.join([str(x) for x in selected_frame_list]),
                                  ' '.join([str(x) for x in trigger_frame_list])])+ '\n')

              


if __name__ == '__main__':
    main()
