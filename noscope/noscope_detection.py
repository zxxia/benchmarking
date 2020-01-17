

"""VideoStorm Overfitting Script."""
import argparse
import csv
import pdb
import sys
import os
import copy
# from video import YoutubeVideo
import numpy as np
from collections import defaultdict
sys.path.append('../../')
from benchmarking.noscope.NoScope import NoScope, load_ground_truth, \
    load_simple_model_classification, train_small_model
from benchmarking.video import YoutubeVideo
from benchmarking.utils.model_utils import eval_single_image
from benchmarking.utils.utils import interpolation, compute_f1

THRESH_LIST = np.arange(0.7, 1, 0.1)
OFFSET = 0  # The time offset from the start of the video. Unit: seconds


VIDEOS = ['crossroad', 'crossroad2', 'crossroad3', 'crossroad4', 'drift',
          'driving1', 'driving_downtown', 'highway',
          'nyc', 'jp',  'lane_split',  'driving2',
          'motorway', 'park', 'russia', 'russia1', 
          'traffic', 'tw', 'tw1',
          'tw_under_bridge']

# VIDEOS = ['cropped_driving2', 'cropped_crossroad4', 'cropped_crossroad4_2', 'cropped_crossroad5' ]

VIDEOS = ['driving1']
DT_ROOT = '/mnt/data/zhujun/dataset/Youtube'
SHORT_VIDEO_LENGTH = 30

profile_length = 30

def eval_images(image_range, original_video, video, thresh=0.8):
    """Evaluate the tp, fp, fn of a range of images."""
    # sample_rate = round(original_config.fps/target_config.fps)
    tpos = defaultdict(int)
    fpos = defaultdict(int)
    fneg = defaultdict(int)
    save_dt = []
    gtruth = original_video.get_video_detection()
    dets = video.get_video_detection()
    trigger_frame_list = []
    for idx in range(image_range[0], image_range[1] + 1):
        if idx not in dets or idx not in gtruth:
            continue
        current_dt = copy.deepcopy(dets[idx])
        current_gt = copy.deepcopy(gtruth[idx])

        # check if the confidence score meets the threshold
        score_list = [x[5] for x in current_dt]
        if score_list != []:
            if np.min(score_list) < thresh:
                # trigger full model
                dt_final = copy.deepcopy(current_gt)
                trigger_frame += 1
            else:
                dt_final = copy.deepcopy(current_dt)
        else:
            dt_final = copy.deepcopy(current_dt)
        tpos[idx], fpos[idx], fneg[idx] = \
            eval_single_image(current_gt, dt_final)

        # print(idx, tpos[idx], fpos[idx], fneg[idx])
    print('# of triggered frame is ', trigger_frame)
    return sum(tpos.values()), sum(fpos.values()), sum(fneg.values())

def profile()


def main():
    """NoScope."""
    gpu_num = '2'
    
    for name in VIDEOS:
        if "cropped" in name:
            resol = '360p'
        else:
            resol = '720p'

        metadata_file = DT_ROOT + '/{}/metadata.json'.format(name)
        dt_file = os.path.join(
            DT_ROOT, name, resol,
            'profile/updated_gt_FasterRCNN_COCO_no_filter.csv')
        original_video = YoutubeVideo(name, resol, metadata_file, dt_file, None)

        # dt_file = os.path.join(
        #     DT_ROOT, name, resol,
        #     'profile/updated_gt_mobilenet_COCO_no_filter.csv')
        # mobilenet_video = YoutubeVideo(name, resol, metadata_file, dt_file, None)

        dt_file = './' + name + '/data/updated_gt_mobilenetFinetuned_COCO_no_filter.csv'
        new_mobilenet_video = YoutubeVideo(name, resol, metadata_file, dt_file, None)

        num_of_short_videos = original_video.frame_count // (
            SHORT_VIDEO_LENGTH*original_video.frame_rate)

        for i in range(num_of_short_videos):
            clip = name + '_' + str(i)
            start_frame = i*SHORT_VIDEO_LENGTH * \
                original_video.frame_rate+1+OFFSET*original_video.frame_rate
            end_frame = (i+1)*SHORT_VIDEO_LENGTH * \
                original_video.frame_rate+OFFSET*original_video.frame_rate
            print('{} start={} end={}'.format(clip, start_frame, end_frame))

            # tp, fp, fn = eval_images([start_frame, end_frame], original_video, mobilenet_video)
            # f1_score = compute_f1(tp, fp, fn)
            f1_list = []
            for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
                tp, fp, fn = eval_images([start_frame, end_frame], original_video, new_mobilenet_video)
                new_f1_score = compute_f1(tp, fp, fn)      

              


if __name__ == '__main__':
    main()
