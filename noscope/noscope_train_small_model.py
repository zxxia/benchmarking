"""VideoStorm Overfitting Script."""
import argparse
import csv
import pdb
import sys
import os
# from video import YoutubeVideo
import numpy as np
sys.path.append('../../')

from benchmarking.noscope.NoScope import NoScope, load_ground_truth, \
    load_simple_model_classification, train_small_model
from benchmarking.video import YoutubeVideo

THRESH_LIST = np.arange(0.7, 1, 0.1)

OFFSET = 0  # The time offset from the start of the video. Unit: seconds


VIDEOS = ['crossroad', 'crossroad2', 'crossroad3', 'crossroad4','drift',
          'driving1', 'driving_downtown', 'highway',
          'nyc', 'jp',  'lane_split',  'driving2',
          'motorway', 'park', 'russia', 'russia1', 
          'traffic', 'tw', 'tw1',
          'tw_under_bridge']

# VIDEOS = ['cropped_driving2', 'cropped_crossroad4', 'cropped_crossroad4_2', 'cropped_crossroad5' ]
VIDEOS = ['crossroad3', 'crossroad4', 'motorway', 'drift'
]
VIDEOS = ['cropped_driving2']
# DT_ROOT = '/data/zxxia/benchmarking/results/videos'
DT_ROOT = '/mnt/data/zhujun/dataset/Youtube'
SHORT_VIDEO_LENGTH = 30

profile_length = 30

def main():
    """NoScope."""
    gpu_num = '1'
    triggered_frames = []
    # tstamp = 0
    for name in VIDEOS:
        if "cropped" in name:
            resol = '360p'
        else:
            resol = '720p'

        pipeline = NoScope(THRESH_LIST, 'noscope_profile_{}.csv'.format(name))
        metadata_file = DT_ROOT + '/{}/metadata.json'.format(name)
        model = 'FasterRCNN'
        dt_file = os.path.join(
            DT_ROOT, name, resol,
            'profile/updated_gt_FasterRCNN_COCO_no_filter.csv')
        img_path = os.path.join(
            DT_ROOT, name, resol)
        video = YoutubeVideo(name, resol, metadata_file, dt_file, img_path)
        model_save_path = './trained_models/'
        train_small_model(name, gpu_num, model_save_path, video)


if __name__ == '__main__':
    main()
