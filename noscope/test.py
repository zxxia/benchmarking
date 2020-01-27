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

OFFSET = 0  # The time offset from the start of the video. Unit: seconds

DT_ROOT = '/mnt/data/zhujun/dataset/Youtube'
SHORT_VIDEO_LENGTH = 30
profile_length = 10
def main():
    """NoScope."""
    for name in ['crossroad2']:
        resol = '720p'
        metadata_file = DT_ROOT + '/{}/metadata.json'.format(name)
        img_path = os.path.join('/mnt/data/zhujun/dataset/Vigil_result/blackbg/', name)
        # img_path = os.path.join(DT_ROOT, name, resol)
        dt_file = os.path.join(
            DT_ROOT, name, resol,
            'profile/updated_gt_FasterRCNN_COCO_no_filter.csv')
        original_video = YoutubeVideo(name, resol, metadata_file, dt_file, img_path)


        num_of_short_videos = original_video.frame_count // (
            SHORT_VIDEO_LENGTH*original_video.frame_rate)


        # profile on first segment 
        i = 0
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


        test_start = profile_end + 1
        test_end = end_frame

        original_bw = original_video.encode(
        os.path.join(clip+'.mp4'), 
        list(range(test_start, test_end)),
        original_video.frame_rate, save_video=True,crf=25)







              


if __name__ == '__main__':
    main()
