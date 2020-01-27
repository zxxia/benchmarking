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


# VIDEOS = ['crossroad', 'crossroad2', 'crossroad3', 'crossroad4','drift',
#           'driving1', 'driving_downtown', 'highway',
#           'nyc', 'jp',  'lane_split',  'driving2',
#           'motorway', 'park', 'russia', 'russia1', 
#           'traffic', 'tw', 'tw1',
#           'tw_under_bridge']

# VIDEOS = ['cropped_driving2', 'cropped_crossroad4', 'cropped_crossroad4_2', 'cropped_crossroad5' ]
# VIDEOS = ['crossroad3', 'crossroad4', 'motorway', 'drift'
# ]
# DT_ROOT = '/data/zxxia/benchmarking/results/videos'
DT_ROOT = '/mnt/data/zhujun/dataset/Youtube'
SHORT_VIDEO_LENGTH = 30
OFFSET = 0

VIDEOS = [x for x in os.listdir(DT_ROOT) if os.path.isdir(os.path.join(DT_ROOT, x))]

profile_length = 30
model_list = ['mobilenet', 'Inception', 'FasterRCNN50', 'FasterRCNN']
# model_list = ['FasterRCNN50']

def compute_easy_frame_percentage(original_video, frame_range, large_box_thresh=0.2):
    gt = original_video.get_video_classification_label()
    easy_frame_cn = 0
    cn = 0
    for frame_idx in range(frame_range[0], frame_range[1]):
        label = gt[frame_idx]
        cn += 1
        if label[0] == 'no_object' or label[1] > large_box_thresh:
            easy_frame_cn += 1
    return float(easy_frame_cn)/cn

f_out = open('./results/model_selection_overfitting_mergelabel.csv', 'w')
f_out.write('dataset,best_model,f1,gpu,easy_frame_percentage\n')
for name in VIDEOS:
    if "cropped" in name:
        resol = '360p'
    else:
        resol = '720p'

    pipeline = ModelSelection(model_list, './results/' + name + '_profile_log_mergelabel.csv')
    metadata_file = DT_ROOT + '/{}/metadata.json'.format(name)
    model = 'FasterRCNN'
    dt_file = os.path.join(
        DT_ROOT, name, resol,
        'profile/updated_gt_' + model + '_COCO_no_filter.csv')
    original_video = YoutubeVideo(name, resol, metadata_file, dt_file, None, merge_label_flag=True)
    videos = {}
    dt_all = {}
    for model in model_list:
        dt_file = os.path.join(
            DT_ROOT, name, resol,
            'profile/updated_gt_' + model + '_COCO_no_filter.csv')
        video = YoutubeVideo(name, resol, metadata_file, dt_file, None, merge_label_flag=True)
        videos[model] = video

    # img_path = '/mnt/data/zhujun/dataset/Youtube/' + name + '/360p/' 
    # for frame_idx in range(1, 1000):
    #     dt_current = [dt_all[model][frame_idx][0] for model in model_list]
    #     # print(frame_idx, list(zip(model_list, dt_current)))
    #     img_filename = img_path + format(frame_idx, '06d') + '.jpg'
    #     img = cv2.imread(img_filename)
    #     cv2.putText(img, dt_current[0], (100, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0))
    #     cv2.putText(img, dt_current[1], (100, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
    #     cv2.putText(img, dt_current[2], (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
    #     cv2.putText(img, dt_current[3], (100, 340), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
    #     cv2.imwrite('./labeled_img_'+format(frame_idx, '06d')+'.jpg', img)




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
        profile_start = start_frame
        profile_end = start_frame + original_video.frame_rate * \
            profile_length - 1

        print('profile {} start={} end={}'.format(
            clip, profile_start, profile_end))
        best_model = \
            pipeline.profile(clip, videos, original_video,
                                [profile_start, profile_end])

        print("Profile {}: best model is {}"
                .format(clip, best_model))

        test_start = start_frame
        test_end = end_frame # overfitting setting

        print('Evaluate {} start={} end={}'.format(
            clip, test_start, test_end))
        f1_score, relative_gpu = pipeline.evaluate(original_video,
            videos, best_model,
            [test_start, test_end])

        print('{} best model={} ==> tested f1={}'
                .format(clip,
                        best_model, f1_score))

        easy_frame_percentage = compute_easy_frame_percentage(original_video, 
                                                                [test_start, test_end])
        f_out.write(','.join([clip, best_model,
                                str(f1_score),
                                str(relative_gpu),
                                str(easy_frame_percentage)]) + '\n')        


