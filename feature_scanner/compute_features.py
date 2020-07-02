import csv
import os

import numpy as np
from scipy import stats

from feature_scanner.features import (compute_arrival_rate,
                                      compute_nb_object_per_frame,
                                      compute_percentage_frame_with_new_object,
                                      compute_percentage_frame_with_object,
                                      compute_velocity,
                                      compute_video_object_size,
                                      count_unique_class)
from videos import get_dataset_class, get_seg_paths

HEADER = ['Video_name', 'object count median', 'object count avg',
          'object count mode', 'object count var', 'object count std',
          'object count skewness', 'object count kurtosis',
          'object count second_moment', 'object count percentile10',
          'object count percentile25', 'object count percentile75',
          'object count percentile90', 'object count iqr',
          'object count entropy',

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


def compute_features(args):
    roots = args.root
    dataset_names = len(VIDEOS) * ['youtube'] + ['mot16']
    video_names = VIDEOS + [None]

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
                video = dataset_class(
                    seg_path, name, '720p', 'faster_rcnn_resnet101',
                    filter_flag=True, classes_interested=classes_interested)
                # load full model's detection results as ground truth
                gt = video.get_video_detection()
                velocity = compute_velocity(
                    gt, video.start_frame_index, video.end_frame_index,
                    video.frame_rate)
                arrival_rate = compute_arrival_rate(
                    gt, video.start_frame_index, video.end_frame_index,
                    video.frame_rate)
                obj_size, tot_obj_size = compute_video_object_size(
                    gt, video.start_frame_index, video.end_frame_index,
                    video.resolution)
                nb_object = compute_nb_object_per_frame(
                    gt, video.start_frame_index, video.end_frame_index)

                chunk_frame_cnt = args.short_video_length * video.frame_rate
                nb_chunks = video.frame_count // chunk_frame_cnt
                if nb_chunks == 0:
                    nb_chunks = 1

                for i in range(nb_chunks):

                    clip = seg_name + '_' + str(i)
                    start_frame = i * chunk_frame_cnt + video.start_frame_index
                    end_frame = (i + 1) * chunk_frame_cnt
                    end_frame = min((i + 1) * chunk_frame_cnt,
                                    video.end_frame_index)
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
                        output_row = [clip] + [None] * (len(HEADER) - 1)
                        writer.writerow(output_row)
