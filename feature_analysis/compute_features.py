import csv
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

from feature_analysis.features import (
    compute_arrival_rate, compute_nb_object_per_frame,
    compute_percentage_frame_with_new_object,
    compute_percentage_frame_with_object, compute_velocity,
    compute_video_object_size, count_unique_class)
from utils.utils import load_full_model_detection
from videos import get_dataset_class, get_seg_paths

VIDEOS = ['crossroad', 'crossroad2', 'crossroad3', 'crossroad4', 'drift',
          'driving1', 'driving_downtown', 'highway',
          'nyc', 'jp',  'lane_split',  'driving2',
          'motorway', 'park', 'russia', 'russia1',
          'traffic', 'tw', 'tw1', 'tw_under_bridge']
VIDEOS = ['chicago_virtual_run', 'tv_show', 'london']
DT_ROOT = '/data/zxxia/benchmarking/results/videos'
# DT_ROOT = '/mnt/data/zhujun/dataset/Youtube'
SHORT_VIDEO_LENGTH = 30
ROOT = '/data/zxxia/videos'

HEADER = ['Video_name', 'object count median', 'object count avg',
          'object count mode',
          'object count var', 'object count std', 'object count skewness',
          'object count kurtosis', 'object count second_moment',
          'object count percentile10', 'object count percentile25',
          'object count percentile75', 'object count percentile90',
          'object count iqr', 'object count entropy',

          'object area median', 'object area avg', 'object area mode',
          'object area var', 'object area std', 'object area skewness',
          'object area kurtosis', 'object area second_moment',
          'object area percentile10', 'object area percentile25',
          'object area percentile75', 'object area percentile90',
          'object area iqr', 'object area entropy',

          'arrival rate median', 'arrival rate avg', 'arrival rate mode',
          'arrival rate var', 'arrival rate std', 'arrival rate skewness',
          'arrival rate kurtosis', 'arrival rate second_moment',
          'arrival rate percentile10', 'arrival rate percentile25',
          'arrival rate percentile75', 'arrival rate percentile90',
          'arrival rate iqr', 'arrival rate entropy',

          'velocity median', 'velocity avg', 'velocity mode', 'velocity var',
          'velocity std', 'velocity skewness', 'velocity kurtosis',
          'velocity second_moment', 'velocity percentile10',
          'velocity percentile25', 'velocity percentile75',
          'velocity percentile90', 'velocity iqr', 'velocity entropy',

          'total area median', 'total area avg', 'total area mode',
          'total area var', 'total area std', 'total area skewness',
          'total area kurtosis', 'total area second_moment',
          'total area percentile10', 'total area percentile25',
          'total area percentile75', 'total area percentile90',
          'total area iqr', 'total area entropy',
          'percent_of_frame_w_object', 'percent_of_frame_w_new_object',
          'nb_distinct_classes']

# OUTPUT_FILE = 'video_features_{}s/allvideo_features_long_add_width_20_filter.csv'.format(SHORT_VIDEO_LENGTH)
OUTPUT_FILE = f'person_videos/person_video_features_{SHORT_VIDEO_LENGTH}.csv'

def feature_gen(data):
    if data == []:
        return list(np.zeros((1, 14))[0])
    median = np.median(data)
    avg = np.mean(data)
    mode = stats.mode(data)[0][0]
    var = np.var(data)
    std = np.std(data)
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    second_moment = stats.moment(data, moment=2)
    percentile10 = np.percentile(data, 10)
    percentile25 = np.percentile(data, 25)
    percentile75 = np.percentile(data, 75)
    percentile90 = np.percentile(data, 90)
    iqr = percentile75 - percentile25
    entropy = stats.entropy(data)
    statistics = [median, avg, mode, var, std, skewness, kurtosis,
                  second_moment, percentile10, percentile25, percentile75,
                  percentile90, iqr, entropy]
    return statistics

roots = len(VIDEOS) * [ROOT] + ['/data/zxxia/MOT16']
dataset_names = len(VIDEOS) * ['youtube'] + ['mot16']
video_names  = VIDEOS + [None]

with open(OUTPUT_FILE, 'w', 1) as f:
    writer = csv.writer(f)
    writer.writerow(HEADER)
    for root, name, dataset_name in zip(roots, video_names, dataset_names):
        print(name)
        classes_interested = {1}
        dataset_class = get_dataset_class(dataset_name)
        seg_paths = get_seg_paths(root, dataset_name, name)
        for seg_path in seg_paths:
            seg_name = os.path.basename(seg_path)
            video = dataset_class(seg_path, name, '720p',
                                  'faster_rcnn_resnet101', filter_flag=True,
                                  classes_interested=classes_interested)
            # load full model's detection results as ground truth
            gt = video.get_video_detection()
            velocity = compute_velocity(gt, video.start_frame_index,
                                        video.end_frame_index, video.frame_rate)
            arrival_rate = compute_arrival_rate(gt, video.start_frame_index,
                                                video.end_frame_index,
                                                video.frame_rate)
            obj_size, tot_obj_size = compute_video_object_size(
                gt, video.start_frame_index, video.end_frame_index,
                video.resolution)
            nb_object = compute_nb_object_per_frame(gt, video.start_frame_index,
                                                    video.end_frame_index)

            chunk_frame_cnt = SHORT_VIDEO_LENGTH * video.frame_rate
            nb_chunks = video.frame_count // chunk_frame_cnt
            if nb_chunks == 0:
                nb_chunks = 1

            for i in range(nb_chunks):

                clip = seg_name + '_' + str(i)
                start_frame = i * chunk_frame_cnt + video.start_frame_index
                end_frame = (i + 1) * chunk_frame_cnt
                end_frame = min((i + 1) * chunk_frame_cnt, video.end_frame_index)
                features = []

                velo = []
                arr_rate = []
                sizes = []
                tot_sizes = []
                nb_obj = []
                percent_with_obj = compute_percentage_frame_with_object(
                    gt, start_frame, end_frame)
                percent_with_new_obj = compute_percentage_frame_with_new_object(
                    gt, start_frame, end_frame)
                nb_distinct_classes = count_unique_class(
                    video.get_video_detection(), start_frame, end_frame)

                for j in range(start_frame, end_frame+1):
                    velo.extend(velocity[j])
                    arr_rate.append(arrival_rate[j])
                    sizes.extend(obj_size[j])
                    tot_sizes.append(tot_obj_size[j])
                    nb_obj.append(nb_object[j])
                if velo and arr_rate and sizes and tot_sizes and nb_obj:
                    features += feature_gen(nb_obj)
                    features += feature_gen(sizes)
                    features += feature_gen(arr_rate)
                    features += feature_gen(velo)
                    features += feature_gen(tot_sizes)
                    features.append(percent_with_obj)
                    features.append(percent_with_new_obj)
                    features.append(nb_distinct_classes)
                    output_row = [clip] + features
                    writer.writerow(output_row)
                else:
                    output_row = [clip] + [None] * 73
                    writer.writerow(output_row)


# WAYMO_ROOT = '/data/zxxia/ekya/datasets/waymo_images'
# with open('video_features_{}s/waymovideo_features_long_add_width_20_filter.csv'.format(SHORT_VIDEO_LENGTH), 'w', 1) as f:
#     writer = csv.writer(f)
#     writer.writerow(
#         ['Video_name', 'object count median', 'object count avg',
#          'object count mode',
#          'object count var', 'object count std', 'object count skewness',
#          'object count kurtosis', 'object count second_moment',
#          'object count percentile10', 'object count percentile25',
#          'object count percentile75', 'object count percentile90',
#          'object count iqr', 'object count entropy',
#
#          'object area median', 'object area avg', 'object area mode',
#          'object area var', 'object area std', 'object area skewness',
#          'object area kurtosis', 'object area second_moment',
#          'object area percentile10', 'object area percentile25',
#          'object area percentile75', 'object area percentile90',
#          'object area iqr', 'object area entropy',
#
#          'arrival rate median', 'arrival rate avg', 'arrival rate mode',
#          'arrival rate var', 'arrival rate std', 'arrival rate skewness',
#          'arrival rate kurtosis', 'arrival rate second_moment',
#          'arrival rate percentile10', 'arrival rate percentile25',
#          'arrival rate percentile75', 'arrival rate percentile90',
#          'arrival rate iqr', 'arrival rate entropy',
#
#          'velocity median', 'velocity avg', 'velocity mode', 'velocity var',
#          'velocity std', 'velocity skewness', 'velocity kurtosis',
#          'velocity second_moment', 'velocity percentile10',
#          'velocity percentile25', 'velocity percentile75',
#          'velocity percentile90', 'velocity iqr', 'velocity entropy',
#
#          'total area median', 'total area avg', 'total area mode',
#          'total area var', 'total area std', 'total area skewness',
#          'total area kurtosis', 'total area second_moment',
#          'total area percentile10', 'total area percentile25',
#          'total area percentile75', 'total area percentile90',
#          'total area iqr', 'total area entropy',
#          'percent_of_frame_w_object', 'percent_of_frame_w_new_object',
#          'nb_distinct_classes'])
#     for seg_path in glob.glob(os.path.join(WAYMO_ROOT, '*')):
#         name = os.path.basename(seg_path)
#         print(name)
#         dt_file = os.path.join(seg_path, 'FRONT/profile',
#                                'updated_gt_FasterRCNN_COCO_no_filter.csv')
#         video = WaymoVideo(name, '720p', dt_file, None)
#         # load full model's detection results as ground truth
#         gt = video.get_video_detection()
#         velocity = compute_velocity(gt, video.start_frame_index,
#                                     video.end_frame_index, video.frame_rate)
#         arrival_rate = compute_arrival_rate(gt, video.start_frame_index,
#                                             video.end_frame_index,
#                                             video.frame_rate)
#         obj_size, tot_obj_size = compute_video_object_size(
#             gt, video.start_frame_index, video.end_frame_index,
#             video.resolution)
#         nb_object = compute_nb_object_per_frame(gt, video.start_frame_index,
#                                                 video.end_frame_index)
#
#         features = []
#         velo = []
#         arr_rate = []
#         sizes = []
#         tot_sizes = []
#         nb_obj = []
#         percent_with_obj = compute_percentage_frame_with_object(
#             gt, video.start_frame_index, video.end_frame_index)
#         percent_with_new_obj = compute_percentage_frame_with_new_object(
#             gt, video.start_frame_index, video.end_frame_index)
#         nb_distinct_classes = count_unique_class(
#             video.get_video_detection(), video.start_frame_index,
#             video.end_frame_index)
#
#         for j in range(video.start_frame_index, video.end_frame_index+1):
#             velo.extend(velocity[j])
#             arr_rate.append(arrival_rate[j])
#             sizes.extend(obj_size[j])
#             tot_sizes.append(tot_obj_size[j])
#             nb_obj.append(nb_object[j])
#         if velo and arr_rate and tot_sizes and nb_obj:
#             features += feature_gen(nb_obj)
#             features += feature_gen(sizes)
#             features += feature_gen(arr_rate)
#             features += feature_gen(velo)
#             features += feature_gen(tot_sizes)
#             features.append(percent_with_obj)
#             features.append(percent_with_new_obj)
#             features.append(nb_distinct_classes)
#             output_row = [name] + features
#             writer.writerow(output_row)

# VIDEOS = ['cropped_crossroad4_2', 'cropped_crossroad4', 'cropped_crossroad5',
#           'cropped_driving2']
# with open(
#     'video_features_{}s/noscopevideo_features_long_add_width_20_filter.csv'
#         .format(SHORT_VIDEO_LENGTH), 'w', 1) as f:
#     writer = csv.writer(f)
#     writer.writerow(
#         ['Video_name', 'object count median', 'object count avg',
#          'object count mode',
#          'object count var', 'object count std', 'object count skewness',
#          'object count kurtosis', 'object count second_moment',
#          'object count percentile10', 'object count percentile25',
#          'object count percentile75', 'object count percentile90',
#          'object count iqr', 'object count entropy',
#
#          'object area median', 'object area avg', 'object area mode',
#          'object area var', 'object area std', 'object area skewness',
#          'object area kurtosis', 'object area second_moment',
#          'object area percentile10', 'object area percentile25',
#          'object area percentile75', 'object area percentile90',
#          'object area iqr', 'object area entropy',
#
#          'arrival rate median', 'arrival rate avg', 'arrival rate mode',
#          'arrival rate var', 'arrival rate std', 'arrival rate skewness',
#          'arrival rate kurtosis', 'arrival rate second_moment',
#          'arrival rate percentile10', 'arrival rate percentile25',
#          'arrival rate percentile75', 'arrival rate percentile90',
#          'arrival rate iqr', 'arrival rate entropy',
#
#          'velocity median', 'velocity avg', 'velocity mode', 'velocity var',
#          'velocity std', 'velocity skewness', 'velocity kurtosis',
#          'velocity second_moment', 'velocity percentile10',
#          'velocity percentile25', 'velocity percentile75',
#          'velocity percentile90', 'velocity iqr', 'velocity entropy',
#
#          'total area median', 'total area avg', 'total area mode',
#          'total area var', 'total area std', 'total area skewness',
#          'total area kurtosis', 'total area second_moment',
#          'total area percentile10', 'total area percentile25',
#          'total area percentile75', 'total area percentile90',
#          'total area iqr', 'total area entropy',
#          'percent_of_frame_w_object', 'percent_of_frame_w_new_object',
#          'nb_distinct_classes', 'similarity'])
#     for name in VIDEOS:
#         print(name)
#         dt_file = os.path.join(DT_ROOT, name,
#                                'updated_gt_FasterRCNN_COCO_no_filter.csv')
#         similarity = pd.read_csv(
#             '/home/zxxia/Projects/benchmarking/model_pruning/{}_similarity_simple_greyscale_normalized_3.csv'.format(name))
#         # metadata_file = DT_ROOT + '/{}/metadata.json'.format(name)
#         # metadata_file = '/data/zxxia/videos/{}/metadata.json'.format(name)
#         # video = YoutubeVideo(name, '720p', None, dt_file, None)
#         dets, frame_count = load_full_model_detection(dt_file)
#         # load full model's detection results as ground truth
#         gt = dets
#         start_frame_index = min(gt)
#         end_frame_index = max(gt)
#         frame_rate = 30
#         resolution = (600, 400)
#         frame_count
#         velocity = compute_velocity(
#             gt, start_frame_index, end_frame_index, frame_rate)
#         arrival_rate = compute_arrival_rate(gt, start_frame_index,
#                                             end_frame_index, frame_rate)
#         obj_size, tot_obj_size = compute_video_object_size(
#             gt, start_frame_index, end_frame_index,
#             resolution)
#         nb_object = compute_nb_object_per_frame(gt, start_frame_index,
#                                                 end_frame_index)
#
#         chunk_frame_cnt = SHORT_VIDEO_LENGTH * frame_rate
#         nb_chunks = frame_count // chunk_frame_cnt
#
#         for i in range(nb_chunks):
#
#             clip = name + '_' + str(i)
#             start_frame = i * chunk_frame_cnt + start_frame_index
#             end_frame = (i + 1) * chunk_frame_cnt
#             features = []
#
#             velo = []
#             arr_rate = []
#             sizes = []
#             tot_sizes = []
#             nb_obj = []
#             percent_with_obj = compute_percentage_frame_with_object(
#                 gt, start_frame, end_frame)
#             percent_with_new_obj = compute_percentage_frame_with_new_object(
#                 gt, start_frame, end_frame)
#             nb_distinct_classes = count_unique_class(
#                 gt, start_frame, end_frame)
#
#             for j in range(start_frame, end_frame+1):
#                 velo.extend(velocity[j])
#                 arr_rate.append(arrival_rate[j])
#                 sizes.extend(obj_size[j])
#                 tot_sizes.append(tot_obj_size[j])
#                 nb_obj.append(nb_object[j])
#             if velo and arr_rate and sizes and tot_sizes and nb_obj:
#                 features += feature_gen(nb_obj)
#                 features += feature_gen(sizes)
#                 features += feature_gen(arr_rate)
#                 features += feature_gen(velo)
#                 features += feature_gen(tot_sizes)
#                 features.append(percent_with_obj)
#                 features.append(percent_with_new_obj)
#                 features.append(nb_distinct_classes)
#                 if i < similarity.shape[0]:
#                     # import pdb
#                     # pdb.set_trace()
#                     features.append(similarity['similarity'].iloc[i])
#                 else:
#                     features.append(np.nan)
#                 output_row = [clip] + features
#                 writer.writerow(output_row)
