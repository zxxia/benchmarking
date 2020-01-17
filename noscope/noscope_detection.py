"""NoScope Overfitting Script."""
import argparse
import csv
import pdb
import sys
import os
import copy
import numpy as np
from collections import defaultdict
sys.path.append('../../')
from benchmarking.noscope.Noscope import NoScope
from benchmarking.video import YoutubeVideo

THRESH_LIST = np.arange(0.3, 1.1, 0.1)
MSE_list = [0, 50, 100, 200]
OFFSET = 0  # The time offset from the start of the video. Unit: seconds
VIDEOS = ['crossroad', 'crossroad2', 'crossroad3', 'crossroad4', 'drift',
          'driving1', 'driving_downtown', 'highway',
          'nyc', 'jp',  'lane_split',  'driving2',
          'motorway', 'park', 'russia', 'russia1', 
          'traffic', 'tw', 'tw1',
          'tw_under_bridge']

VIDEOS = ['crossroad4']
DT_ROOT = '/mnt/data/zhujun/dataset/Youtube'
SHORT_VIDEO_LENGTH = 30
profile_length = 10
SMALL_MODEL_PATH = '/mnt/data/zhujun/dataset/NoScope_finetuned_models'

def main():
    """NoScope."""
    for name in VIDEOS:
        if "cropped" in name:
            resol = '360p'
        else:
            resol = '720p'
        OUTPUT_PATH = os.path.join(
            SMALL_MODEL_PATH, name, 'data'
        )
        pipeline = NoScope(THRESH_LIST, MSE_list, OUTPUT_PATH + '/tmp_log.csv')
        f_out = open('Noscope_e2e_result_' + name + '_with_frame_diff.csv', 'w')
        f_out.write('dataset,best_confidence_score_thresh,f1,bandwidth, triggered_frames\n')
        metadata_file = DT_ROOT + '/{}/metadata.json'.format(name)
        img_path = os.path.join(DT_ROOT, name, resol)
        dt_file = os.path.join(
            DT_ROOT, name, resol,
            'profile/updated_gt_FasterRCNN_COCO_no_filter.csv')
        original_video = YoutubeVideo(name, resol, metadata_file, dt_file, img_path)

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
            profile_end = start_frame + original_video.frame_rate * \
                profile_length - 1

            print('profile {} start={} end={}'.format(
                clip, profile_start, profile_end))
            best_mse_thresh, best_thresh, best_relative_bw = \
                pipeline.profile(clip, original_video, new_mobilenet_video,
                                 [profile_start, profile_end])

            print("Profile {}: best mse thresh={}, best thresh={}, best bw={}"
                  .format(clip, best_mse_thresh, best_thresh, best_relative_bw))

            test_start = profile_end + 1
            test_end = end_frame

            print('Evaluate {} start={} end={}'.format(
                clip, test_start, test_end))
            f1_score, relative_bw, trigger_frame_list = pipeline.evaluate(
                os.path.join(OUTPUT_PATH, clip + '.mp4'), original_video,
                new_mobilenet_video, best_mse_thresh, best_thresh,
                [test_start, test_end])

            print('{} best thresh={} ==> tested f1={}'
                  .format(clip, best_thresh, f1_score))
            f_out.write(','.join([clip, str(best_thresh),
                                  str(f1_score),
                                  str(relative_bw),
                                  ' '.join([str(x) for x in trigger_frame_list])])+ '\n')

              


if __name__ == '__main__':
    main()
