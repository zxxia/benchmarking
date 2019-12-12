# video selection
# import os
import glob
import numpy as np
from collections import defaultdict
import random
import matplotlib.pyplot as plt
from helpers import load_glimpse_results, load_videostorm_profile, \
    load_videostorm_results, load_glimpse_profile


all_feature_names = [
    'object_cn_median', 'object_cn_avg', 'object_cn_mode', 'object_cn_var',
    'object_cn_std',
    'object_cn_skewness', 'object_cn_kurtosis', 'object_cn_second_moment',
    'object_cn_percentile5',
    'object_cn_percentile25', 'object_cn_percentile75',
    'object_cn_percentile95', 'object_cn_iqr',
    'object_cn_entropy',
    'object_size_median', 'object_size_avg', 'object_size_mode',
    'object_size_var', 'object_size_std', 'object_size_skewness',
    'object_size_kurtosis',
    'object_size_second_moment', 'object_size_percentile5',
    'object_size_percentile25',
    'object_size_percentile75', 'object_size_percentile95', 'object_size_iqr',
    'object_size_entropy', 'arrival_rate_median', 'arrival_rate_avg',
    'arrival_rate_mode', 'arrival_rate_var', 'arrival_rate_std',
    'arrival_rate_skewness',
    'arrival_rate_kurtosis', 'arrival_rate_second_moment',
    'arrival_rate_percentile5', 'arrival_rate_percentile25',
    'arrival_rate_percentile75', 'arrival_rate_percentile95',
    'arrival_rate_iqr',
    'arrival_rate_entropy', 'velocity_median', 'velocity_avg', 'velocity_mode',
    'velocity_var', 'velocity_std', 'velocity_skewness', 'velocity_kurtosis',
    'velocity_second_moment', 'velocity_percentile5', 'velocity_percentile25',
    'velocity_percentile75', 'velocity_percentile95', 'velocity_iqr',
    'velocity_entropy', 'total_area_median', 'total_area_avg',
    'total_area_mode',
    'total_area_var', 'total_area_std', 'total_area_skewness',
    'total_area_kurtosis',
    'total_area_second_moment', 'total_area_percentile5',
    'total_area_percentile25',
    'total_area_percentile75', 'total_area_percentile95', 'total_area_iqr',
    'total_area_entropy', 'percentage', 'percentage_w_new_object']


minmax = {}
# 'velocity_avg' and 'percentage' are used in temporal sampling

# 'object_size_avg', 'total_area_avg' are used in spatial sampling
feature_names = ['velocity_avg', 'percentage']
# feature_names = ['object_size_avg', 'total_area_avg']
minmax['velocity_avg'] = (1, 4)
minmax['percentage'] = (0, 1)


def read_feature(feature_file):
    features = {}
    with open(feature_file) as f:
        f.readline()
        for line in f:
            line_list = line.strip().split(',')
            key = line_list[0]
            features[key] = [float(x) for x in line_list[1:]]
    return features


# for dataset with bad lighting, remove those datasets
def filter_bad_dataset(bad_dataset, features):
    filtered_features = {}
    for key in sorted(features.keys()):
        datasetname = '_'.join(key.split('_')[0:-1])
        if datasetname in bad_dataset:
            continue
        else:
            filtered_features[key] = features[key]
    return filtered_features


def main():
    # feature_file = 'features_all_object_type_filter.csv'
    feature_file = 'features_all_type_width_height_filter.csv'
    features = read_feature(feature_file)
    bad_dataset = ['tw_under_bridge', 'russia', 'lane_split']
    filtered_features = filter_bad_dataset(bad_dataset, features)
    print('total num of videos:', len(filtered_features))
    feature_bins = defaultdict(list)

    # for each target feature, first compute its total range
    # then divide the range into 3 buckets (maybe change 3 to variable)
    for feature_name in feature_names:
        feature_index = all_feature_names.index(feature_name)
        feature_list = [filtered_features[key][feature_index]
                        for key in filtered_features]

        feature_range = (np.percentile(feature_list, 5),
                         np.percentile(feature_list, 98))
        if feature_name in minmax:
            feature_bins[feature_name].append(minmax[feature_name][0])
            feature_bins[feature_name].append(
                feature_range[0] + (feature_range[1]-feature_range[0])/5)
            feature_bins[feature_name].append(feature_range[0] +
                                              2*(feature_range[1]-feature_range[0])/5)
            feature_bins[feature_name].append(feature_range[0] +
                                              3*(feature_range[1]-feature_range[0])/5)
            feature_bins[feature_name].append(feature_range[0] +
                                              4*(feature_range[1]-feature_range[0])/5)
            feature_bins[feature_name].append(minmax[feature_name][1])
        else:
            feature_bins[feature_name].append(min(feature_list))
            feature_bins[feature_name].append(
                feature_range[0] + (feature_range[1]-feature_range[0])/5)
            feature_bins[feature_name].append(
                feature_range[0] + 2*(feature_range[1] - feature_range[0])/5)
            feature_bins[feature_name].append(
                feature_range[0] + 3*(feature_range[1] - feature_range[0])/5)
            feature_bins[feature_name].append(
                feature_range[0] + 4*(feature_range[1] - feature_range[0])/5)
            feature_bins[feature_name].append(max(feature_list))

    # assign each video to those bins based on its feature value
    video_assignment = defaultdict(list)
    for key in filtered_features:
        # assign stores the tuple of bin index of all target features
        # e.g. (1, 2) means the first feature falls into the second bin
        # and the second feature falls into the third bin
        assign = []
        for name in feature_names:
            f_v = filtered_features[key][all_feature_names.index(name)]
            if f_v < feature_bins[name][0] or f_v > feature_bins[name][-1]:
                break
            else:
                assign.append(next(x[0] for x in enumerate(
                    feature_bins[name]) if x[1] >= f_v))
        video_assignment[tuple(assign)].append(key)

    object_size_avg_idx = all_feature_names.index('object_size_avg')
    object_size_avg_range = np.ptp(
        [filtered_features[key][object_size_avg_idx] for key in filtered_features])
    tot_area_avg_idx = all_feature_names.index('total_area_avg')
    tot_area_avg_range = np.ptp(
        [filtered_features[key][tot_area_avg_idx] for key in filtered_features])
    # print(object_size_avg_range)
    # print(tot_area_avg_range)
    assigned_videos = []

    for key in video_assignment:
        assigned_videos.extend(video_assignment[key])

    for val in feature_bins['velocity_avg']:
        plt.axvline(x=val, c='k', linestyle='--')
    for val in feature_bins['percentage']:
        plt.axhline(y=val, c='k', linestyle='--')
    velocity_avg_idx = all_feature_names.index('velocity_avg')
    percentage_idx = all_feature_names.index('percentage')

    ft1_to_plot = [filtered_features[video][velocity_avg_idx]
                   for video in assigned_videos]

    ft2_to_plot = [filtered_features[video][percentage_idx]
                   for video in assigned_videos]
    plt.scatter(ft1_to_plot, ft2_to_plot, s=10)
    plt.xlabel('object velocity average')
    plt.ylabel('percetage of frames with object')
    plt.title('before control other featuers')
    plt.xlim(1, 4.1)
    plt.ylim(0, 1.1)
    # ax = plt.gca()
    # ax.set_scale('log')
    axs = plt.gca()
    axs.set_xscale('log')
    axs.set_yscale('log')

    results = []
    max_result = []
    result_len = 0
    for video0 in assigned_videos:
        result = list()
        for video1 in assigned_videos:
            if video0 == video1:
                continue
            if np.abs(filtered_features[video0][object_size_avg_idx] -
                      filtered_features[video1][object_size_avg_idx]) <= 0.05 * object_size_avg_range and \
                np.abs(filtered_features[video0][tot_area_avg_idx] -
                       filtered_features[video1][tot_area_avg_idx]) <= 0.05 * tot_area_avg_range:
                result.append(video1)
        if len(result) > 50:
            results.append(result)
        if len(result) > result_len:
            max_result = result
    # print(len(results))

    for key in video_assignment:
        assigned_videos = video_assignment[key]
        tmp = []
        for video in assigned_videos:
            if video in max_result:
                tmp.append(video)
        video_assignment[key] = tmp

    selected_video = []
    for key in sorted(video_assignment):
        if key == ():
            continue
        if not video_assignment[key]:
            continue
        tmp = random.choice(video_assignment[key])
        print(key, len(video_assignment[key]), tmp)
        selected_video.append(tmp)
    print(selected_video)

    # feature_names = ['velocity_avg', 'percentage']
    velocity_avg_idx = all_feature_names.index('velocity_avg')
    percentage_idx = all_feature_names.index('percentage')

    assigned_videos = []
    for key in video_assignment:
        assigned_videos.extend(video_assignment[key])

    ft1_to_plot = [filtered_features[video][velocity_avg_idx]
                   for video in assigned_videos]

    ft2_to_plot = [filtered_features[video][percentage_idx]
                   for video in assigned_videos]

    selected_ft1_to_plot = [filtered_features[video][velocity_avg_idx]
                            for video in selected_video]
    selected_ft2_to_plot = [filtered_features[video][percentage_idx]
                            for video in selected_video]

    plt.figure()
    for val in feature_bins['velocity_avg']:
        plt.axvline(x=val, c='k', linestyle='--')
    for val in feature_bins['percentage']:
        plt.axhline(y=val, c='k', linestyle='--')
    plt.scatter(ft1_to_plot, ft2_to_plot, s=10, label='after control')
    plt.scatter(selected_ft1_to_plot,
                selected_ft2_to_plot, s=10, label='randomly selected')
    plt.xlabel('object velocity average')
    plt.ylabel('percetage of frames with object')
    plt.title('after control other featuers')
    plt.xlim(1, 4.1)
    plt.ylim(0, 1.1)
    plt.legend()

    plt.figure()

    perf_to_plot = []
    acc_to_plot = []
    for video in assigned_videos:
        video_name = video.split('_')[0]
        # print(video)
        profile_file = '/data/zxxia/benchmarking/videostorm/coverage_results/videostorm_coverage_results_{}.csv'.format(
            video_name)
        vids, perf_dict, acc_dict = load_videostorm_results(profile_file)
        for vid, perf, acc in zip(vids, perf_dict, acc_dict):
            if vid != video:
                continue
            acc_to_plot.append(acc)
            perf_to_plot.append(perf)
            # perf_to_plot.extend(perf_dict[key])
            # acc_to_plot.extend(acc_dict[key])
    plt.scatter(perf_to_plot, acc_to_plot, s=5, label='assgined videos')

    perf_to_plot = []
    acc_to_plot = []
    for video in selected_video:
        video_name = video.split('_')[0]
        # print(video)
        profile_file = '/data/zxxia/benchmarking/videostorm/coverage_results/videostorm_coverage_results_{}.csv'.format(
            video_name)
        vids, perf_dict, acc_dict = load_videostorm_results(profile_file)
        for vid, perf, acc in zip(vids, perf_dict, acc_dict):
            if vid != video:
                continue
            acc_to_plot.append(acc)
            perf_to_plot.append(perf)
            # perf_to_plot.extend(perf_dict[key])
            # acc_to_plot.extend(acc_dict[key])
    plt.scatter(perf_to_plot, acc_to_plot, s=8,
                c='k', label='selected videos')

    perf_to_plot = []
    acc_to_plot = []
    kitti_profile_files = glob.glob(
        "/data/zxxia/benchmarking/videostorm/kitti_coverage_results/*.csv")
    for kitti_file in kitti_profile_files:
        print(kitti_file)
        vids, perf_dict, acc_dict = load_videostorm_results(kitti_file)
        print(len(vids))
        # for key in vids:
        for perf, acc in zip(perf_dict, acc_dict):
                # if perf == 0.05:
            acc_to_plot.append(acc)
            perf_to_plot.append(perf)
            # perf_to_plot.extend(perf_dict[key])
            # acc_to_plot.extend(acc_dict[key])
    # print(perf_to_plot, acc_to_plot)
    plt.scatter(perf_to_plot, acc_to_plot, s=5, c='r', label='kitti')

    # perf_to_plot = []
    # acc_to_plot = []
    # for video in assigned_videos:
    #     video_name = video.split('_')[0]
    #     # print(video)
    #     profile_file = '/data/zxxia/benchmarking/videostorm/overfitting_profile_10_14/videostorm_overfitting_profile_{}.csv'.format(
    #         video_name)
    #     vids, perf_dict, acc_dict = load_videostorm_profile(profile_file)
    #     for key in vids:
    #         if key != video:
    #             continue
    #         for perf, acc in zip(perf_dict[key], acc_dict[key]):
    #             if perf == 0.05:
    #                 acc_to_plot.append(acc)
    #                 perf_to_plot.append(perf)
    #         # perf_to_plot.extend(perf_dict[key])
    #         # acc_to_plot.extend(acc_dict[key])
    # plt.scatter(perf_to_plot, acc_to_plot, s=5)

    # plot videostorm on kitti dataset
    # perf_to_plot = []
    # acc_to_plot = []
    # kitti_profile_files = glob.glob(
    #     "/data/zxxia/benchmarking/videostorm/kitti_overfitting_profile/*.csv")
    # for kitti_file in kitti_profile_files:
    #     print(kitti_file)
    #     vids, perf_dict, acc_dict = load_videostorm_profile(kitti_file)
    #     print(len(vids))
    #     for key in vids:
    #         for perf, acc in zip(perf_dict[key], acc_dict[key]):
    #             if perf == 0.05:
    #                 acc_to_plot.append(acc)
    #                 perf_to_plot.append(perf)
    #         # perf_to_plot.extend(perf_dict[key])
    #         # acc_to_plot.extend(acc_dict[key])
    # print(perf_to_plot, acc_to_plot)
    # plt.scatter(perf_to_plot, acc_to_plot, s=5, label='kitti')

    # perf_to_plot = []
    # acc_to_plot = []
    # for video in selected_video:
    #     video_name = video.split('_')[0]
    #     profile_file = '/data/zxxia/benchmarking/videostorm/overfitting_profile_10_14/videostorm_overfitting_profile_{}.csv'.format(
    #         video_name)
    #     vids, perf_dict, acc_dict = load_videostorm_profile(profile_file)
    #     for key in vids:
    #         if key != video:
    #             continue
    #         for perf, acc in zip(perf_dict[key], acc_dict[key]):
    #             if perf == 0.05:
    #                 acc_to_plot.append(acc)
    #                 perf_to_plot.append(perf)
    #         # perf_to_plot.extend(perf_dict[key])
    #         # acc_to_plot.extend(acc_dict[key])
    # plt.scatter(perf_to_plot, acc_to_plot, s=5, label='selected')

    plt.title('videostorm performance and accuracy coverage')
    plt.xlabel('performance')
    plt.ylabel('f1')
    plt.legend()

    plt.figure()
    perf_to_plot = []
    acc_to_plot = []
    vid_to_plot = []

    for video in assigned_videos:
        video_name = video.split('_')[0]
        # results_file = '/data/zxxia/benchmarking/glimpse/glimpse_frame_select_results_11_27/glimpse_perfect_tracking_{}.csv'.format(
        #     video_name)
        profile_file = '/data/zxxia/benchmarking/glimpse/glimpse_frame_select_profile_11_27/profile_{}.csv'.format(
            video_name)
        vids, perf_dict, acc_dict, para1_dict = load_glimpse_profile(
            profile_file)
        for key in vids:
            if key != video:
                continue
            for perf, acc, para1 in zip(perf_dict[key], acc_dict[key],
                                        para1_dict[key]):
                if 30 <= para1 <= 33:
                    perf_to_plot.append(perf)
                    acc_to_plot.append(acc)
                    vid_to_plot.append(key)
    # for video in assigned_videos:
    #     video_name = video.split('_')[0]
    #     results_file = '/data/zxxia/benchmarking/glimpse/glimpse_frame_select_results_11_27/glimpse_perfect_tracking_{}.csv'.format(
    #         video_name)
    #     # print(len(load_glimpse_results(results_file)))
    #     vids, perfs, accs = load_glimpse_results(results_file)
    #     for key, perf, acc in zip(vids, perfs, accs):
    #         if key != video:
    #             continue
    #         # print(key, perf, acc)
    #         perf_to_plot.append(perf)
    #         acc_to_plot.append(acc)
    #         vid_to_plot.append(key)
    plt.scatter(perf_to_plot, acc_to_plot, s=5)
    # ax = plt.gca()
    # for perf, acc, vid in zip(perf_to_plot, acc_to_plot, vid_to_plot):
    #     ax.annotate(vid, (perf, acc))
    perf_to_plot = []
    acc_to_plot = []
    for video in selected_video:
        video_name = video.split('_')[0]
        # results_file = '/data/zxxia/benchmarking/glimpse/glimpse_frame_select_results_11_27/glimpse_perfect_tracking_{}.csv'.format(
        #     video_name)
        profile_file = '/data/zxxia/benchmarking/glimpse/glimpse_frame_select_profile_11_27/profile_{}.csv'.format(
            video_name)
        vids, perf_dict, acc_dict, para1_dict = load_glimpse_profile(
            profile_file)
        for key in vids:
            if key != video:
                continue
            for perf, acc, para1 in zip(perf_dict[key], acc_dict[key],
                                        para1_dict[key]):
                if 30 <= para1 <= 33:
                    perf_to_plot.append(perf)
                    acc_to_plot.append(acc)
    # for video in selected_video:
    #     video_name = video.split('_')[0]
    #     results_file = '/data/zxxia/benchmarking/glimpse/glimpse_frame_select_results_11_27/glimpse_perfect_tracking_{}.csv'.format(
    #         video_name)
    #     vids, perfs, accs = load_glimpse_results(results_file)
    #     for key, perf, acc in zip(vids, perfs, accs):
    #         if key != video:
    #             continue
    #         perf_to_plot.append(perf)
    #         acc_to_plot.append(acc)
    plt.scatter(perf_to_plot, acc_to_plot, s=5, label='selected')

    plt.legend()
    plt.title('glimpse performance and accuracy coverage')
    plt.xlabel('performance')
    plt.ylabel('f1')
    plt.show()


if __name__ == '__main__':
    main()
