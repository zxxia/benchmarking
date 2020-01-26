''' This script is to used to compute classification performance (f1, gpu) of model selection.
'''
# from collections import defaultdict
import argparse
import os
import sys
sys.path.append('../../')
import cv2
# import pdb
# from utils.model_utils import filter_video_detections, remove_overlappings

from benchmarking.model_selection.ModelSelection import ModelSelection
from benchmarking.video import YoutubeVideo

DT_ROOT = '/mnt/data/zhujun/dataset/Youtube'
SHORT_VIDEO_LENGTH = 30
OFFSET = 0

VIDEOS = [x for x in os.listdir(DT_ROOT) if os.path.isdir(os.path.join(DT_ROOT, x))]

profile_length = 30
model_list = ['mobilenet', 'Inception', 'FasterRCNN50', 'FasterRCNN']


def compute_easy_frame_percentage(original_video, frame_range, large_box_thresh=0.2):
    # [x, y, x+w, y+h, t, score, obj_id]
    gt = original_video.get_video_detection()
    easy_frame_cn = 0
    cn = 0
    for frame_idx in range(frame_range[0], frame_range[1]):
        current_gt = gt[frame_idx]
        cn += 1
        # if no_object or minimal object size meets a threshold, it is a easy frame
        if len(current_gt) == 0:
            easy_frame_cn += 1
        else:
            area = []
            for box in current_gt:
                w = box[2] - box[0]
                h = box[3] - box[1]
                area.append(float(w*h)/(original_video.resolution[0] * original_video.resolution[1]))            
            if min(area) >= large_box_thresh:
                easy_frame_cn += 1
    return float(easy_frame_cn)/cn

f_out = open('./results/easy_frame_percentage.csv', 'w')
f_out.write('dataset,easy_frame_percentage\n')
for name in VIDEOS:
    if "cropped" in name:
        resol = '360p'
    else:
        resol = '720p'

    metadata_file = DT_ROOT + '/{}/metadata.json'.format(name)
    model = 'FasterRCNN'
    dt_file = os.path.join(
        DT_ROOT, name, resol,
        'profile/updated_gt_' + model + '_COCO_no_filter.csv')
    original_video = YoutubeVideo(name, resol, metadata_file, dt_file, None, merge_label_flag=True)
 

    num_of_short_videos = original_video.frame_count // (
        SHORT_VIDEO_LENGTH*original_video.frame_rate)

    for i in range(num_of_short_videos):
        clip = name + '_' + str(i)
        start_frame = i*SHORT_VIDEO_LENGTH * \
            original_video.frame_rate+1+OFFSET*original_video.frame_rate
        end_frame = (i+1)*SHORT_VIDEO_LENGTH * \
            original_video.frame_rate+OFFSET*original_video.frame_rate
        print('{} start={} end={}'.format(clip, start_frame, end_frame))
        # use 30 seconds video for profiling
        test_start = start_frame
        test_end = end_frame # overfitting setting


        easy_frame_percentage = compute_easy_frame_percentage(original_video, 
                                                                [test_start, test_end])
        f_out.write(','.join([clip, 
                            str(easy_frame_percentage)]) + '\n')        


