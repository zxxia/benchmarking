import csv
import os
import numpy as np
import sys
sys.path.append('../../')
from benchmarking.feature_analysis.features import compute_velocity, \
    compute_video_object_size, compute_arrival_rate, \
    compute_percentage_frame_with_object, count_unique_class, compute_nb_object_per_frame, compute_percentage_frame_with_new_object
from benchmarking.video import YoutubeVideo
from benchmarking.utils.model_utils import load_full_model_detection
from scipy import stats

VIDEOS = ['crossroad', 'crossroad2', 'crossroad3', 'crossroad4','drift',
          'driving1', 'driving_downtown', 'highway',
          'nyc', 'jp',  'lane_split',  'driving2',
          'motorway', 'park', 'russia', 'russia1', 
          'traffic', 'tw', 'tw1',
          'tw_under_bridge']

# DT_ROOT = '/data/zxxia/benchmarking/results/videos'
DT_ROOT = '/mnt/data/zhujun/dataset/Youtube'
SHORT_VIDEO_LENGTH = 30

def feature_gen(data):
    if data == []:
        return list(np.zeros((1,14))[0])
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
    statistics = [median, avg, mode, var, std, skewness, kurtosis, second_moment,
                  percentile10, percentile25, percentile75, percentile90, iqr, entropy]
    return statistics

with open('video_features_{}s/allvideo_features_long_add_width_20_filter.csv'.format(SHORT_VIDEO_LENGTH), 'w', 1) as f:
    writer = csv.writer(f)
    f.write('Video_name,'\
        'object_cn (median), avg, mode, var, std, skewness, kurtosis, second_moment,' \
                'percentile10, percentile25, percentile75, percentile90, iqr, entropy,'\
        'object_area (median), avg, mode, var, std, skewness, kurtosis, second_moment,' \
                'percentile10, percentile25, percentile75, percentile90, iqr, entropy,'\
        'arrival_rate (median),avg, mode, var, std, skewness, kurtosis, second_moment,' \
                'percentile10, percentile25, percentile75, percentile90, iqr, entropy,'\
        'velocity (median), avg, mode, var, std, skewness, kurtosis, second_moment,' \
                'percentile10, percentile25, percentile75, percentile90, iqr, entropy,'\
        'total_area (median), avg, mode, var, std, skewness, kurtosis, second_moment,' \
                'percentile10, percentile25, percentile75, percentile90, iqr, entropy,'\
        'percent_of_frame_w_object, percent_of_frame_w_new_object, nb_distinct_classes\n')
    for name in VIDEOS:
        print(name)
        dt_file = os.path.join(
            DT_ROOT, name, '720p',
            'profile/updated_gt_FasterRCNN_COCO_no_filter.csv')
        metadata_file = DT_ROOT + '/{}/metadata.json'.format(name)
        video = YoutubeVideo(name, '720p', metadata_file, dt_file, None)
        # load full model's detection results as ground truth
        gt = video.get_video_detection()
        velocity = compute_velocity(gt, video.start_frame_index,
            video.end_frame_index, video.frame_rate)
        arrival_rate = compute_arrival_rate(gt, video.start_frame_index,
            video.end_frame_index, video.frame_rate)
        obj_size, tot_obj_size = compute_video_object_size(gt, video.start_frame_index,
            video.end_frame_index, video.resolution)
        nb_object = compute_nb_object_per_frame(gt, video.start_frame_index,
            video.end_frame_index)        

        chunk_frame_cnt = SHORT_VIDEO_LENGTH * video.frame_rate
        nb_chunks = video.frame_count // chunk_frame_cnt

        for i in range(nb_chunks):
            
            clip = name + '_' + str(i)
            start_frame = i * chunk_frame_cnt + video.start_frame_index
            end_frame = (i + 1) * chunk_frame_cnt
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


