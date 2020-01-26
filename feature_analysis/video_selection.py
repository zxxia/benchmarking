"""Video selection code for the paper."""
import csv
import pdb
from itertools import zip_longest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VIDEOS = ['crossroad', 'crossroad2', 'crossroad3', 'crossroad4', 'drift',
          'driving1', 'driving_downtown', 'highway',
          'nyc', 'jp',  'lane_split',  'driving2',
          'motorway', 'park', 'russia', 'russia1', 'traffic', 'tw', 'tw1',
          'tw_under_bridge']

NUM_BINS = 6
NUM_CHOICE = 2
np.random.seed(100)


def control_features(videos, feature1, feature2, ratio=0.05):
    """Control other features."""
    assert videos.shape == feature1.shape
    assert videos.shape == feature2.shape
    feature_controled_video_groups = []
    for i, video_i in enumerate(videos):
        result = list()
        for j, video_j in enumerate(videos):
            if video_i == video_j:
                continue
            if np.abs(feature1[i]-feature1[j]) <= ratio*np.ptp(feature1) and \
                    np.abs(feature2[i]-feature2[j]) <= ratio*np.ptp(feature2):
                result.append(video_j)
        feature_controled_video_groups.append(result)
    lengths = [len(group) for group in feature_controled_video_groups]
    biggest_group_idx = np.argmax(lengths)
    # if len(result) > result_len:
    #     max_result = result
    return feature_controled_video_groups[biggest_group_idx]


def divide_bins(videos, feature1, feature2, num_bins):
    """Divide the features into bins and assign videos to each bins."""
    feature1_bins = np.arange(np.min(feature1), np.max(feature1),
                              np.ptp(feature1)/(num_bins+1))
    feature2_bins = np.arange(np.min(feature2), np.max(feature2),
                              np.ptp(feature2)/(num_bins+1))
    bin_videos = {}
    for i in range(num_bins):
        for j in range(num_bins):
            # videos[]
            kept = np.where((feature1 >= feature1_bins[i]) &
                            (feature1 <= feature1_bins[i+1]) &
                            (feature2 >= feature2_bins[j]) &
                            (feature2 <= feature2_bins[j+1]))
            # kept_videos.append(list(videos[kept]))

            print(i, j, kept)
            bin_videos[(i, j)] = videos[kept].tolist()

    return feature1_bins, feature2_bins, bin_videos


def main():
    # aws_selection()
    vs_selection()
    # no_selection()


def no_selection():
    # load all the features
    features = []
    features.append(pd.read_csv(
        'video_features_30s/allvideo_features_long_add_width_20_filter.csv'))
    # features.append(pd.read_csv(
    #     'video_features_30s/waymovideo_features_long_add_width_20_filter.csv'))
    features = pd.concat(features, ignore_index=True)
    print(features.shape)
    # drop nan values
    features = features.dropna()
    print(features.shape)

    # control other features
    feature_controled_video_groups = control_features(
        features['Video_name'].to_numpy(),
        features['object area avg'].to_numpy(),
        features['total area avg'].to_numpy())

    mask = features['Video_name'].isin(feature_controled_video_groups)
    plt.scatter(features['velocity avg'],
                features['percent_of_frame_w_object'], label='all')
    plt.scatter(features[mask]['velocity avg'],
                features[mask]['percent_of_frame_w_object'],
                label='feature controlled')
    plt.xlabel('object velocity')
    plt.ylabel('percent of frames with object')
    plt.ylim(0, 1.1)
    plt.xlim(1, 3.5)

    # assign feature controlled videos into bins
    feature1_bins, feature2_bins, bin_videos = divide_bins(
        features[mask]['Video_name'].to_numpy(),
        features[mask]['velocity avg'].to_numpy(),
        features[mask]['percent_of_frame_w_object'].to_numpy(), NUM_BINS)
    plt.figure()
    all_bin_videos = []
    for key in bin_videos:
        all_bin_videos.extend(bin_videos[key])
    mask = features['Video_name'].isin(all_bin_videos)
    plt.scatter(features[mask]['velocity avg'],
                features[mask]['percent_of_frame_w_object'],
                label='feature controled videos')
    for val in feature1_bins:
        plt.axvline(x=val, c='k', linestyle='--')
    for val in feature2_bins:
        plt.axhline(y=val, c='k', linestyle='--')

    # randomly pick NUM_CHOICE videos from each bin
    selected_videos = []
    for key in bin_videos:
        if bin_videos[key]:
            selected_videos.extend(
                np.random.choice(bin_videos[key], NUM_CHOICE).tolist())
    mask = features['Video_name'].isin(selected_videos)
    plt.scatter(features[mask]['velocity avg'],
                features[mask]['percent_of_frame_w_object'],
                label='selected')
    plt.ylim(0, 1.1)
    plt.xlim(1, 3.5)
    plt.xlabel('object velocity')
    plt.ylabel('percent of frames with object')
    plt.legend()

    # no_results = []
    results = pd.read_csv(
        '~/Projects/benchmarking/noscope/Noscope_e2e_result.csv')
    # vs_results.append(pd.read_csv(
    #     '~/Projects/benchmarking/videostorm/youtube_overfitting_30s/videostorm_overfitting_results_{}.csv'.format(name)))
    # vs_results.append(pd.read_csv(
    #     '~/Projects/benchmarking/videostorm/waymo_720p_e2e/videostorm_e2e_waymo.csv'))
    # vs_results.append(pd.read_csv(
    #     '~/Projects/benchmarking/videostorm/waymo_overfitting/videostorm_overfitting_waymo.csv'))

    # vs_results=pd.concat(vs_results, ignore_index = True)
    plt.figure()
    mask = results['dataset'].isin(selected_videos)
    plt.scatter(results['gpu'], results['f1'], label='all', s=1)
    plt.scatter(results[mask]['gpu'], results[mask]['f1'], label='selected')
    plt.ylim(0, 1.1)
    plt.xlim(0, 1.1)
    plt.xlabel('gpu')
    plt.ylabel('f1')
    plt.legend()

    kitti_results = pd.read_csv('../noscope/Noscope_e2e_result_KITTI.csv')
    # canda_vs_results = pd.read_csv(
    #     '../videostorm/canada_e2e/videostorm_e2e_canada.csv')
    with open('../plot/no_video_selection.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['coverage video', 'coverage gpu', 'coverage f1',
                         'selected video', 'selected gpu', 'selected f1',
                         'kitti video', 'kitti gpu', 'kitti_f1',
                         'long video', 'long video gpu', 'long video f1'])
        for values in zip_longest(
            results['dataset'], results['gpu'], results['f1'],
            results[mask]['dataset'], results[mask]['gpu'], results[mask]['f1'],
            kitti_results['dataset'], kitti_results['gpu'], kitti_results['f1'],
            results[results['dataset'].str.contains('park')]['dataset'],
                results[results['dataset'].str.contains('park')]['gpu'],
                results[results['dataset'].str.contains('park')]['f1']):
            writer.writerow(values)

    plt.show()


def vs_selection():
    # load all the features
    features = []
    features.append(pd.read_csv(
        'video_features_30s/allvideo_features_long_add_width_20_filter.csv'))
    features.append(pd.read_csv(
        'video_features_30s/waymovideo_features_long_add_width_20_filter.csv'))
    features = pd.concat(features, ignore_index=True)
    print(features.shape)
    # drop nan values
    features = features.dropna()
    print(features.shape)

    # control other features
    feature_controled_video_groups = control_features(
        features['Video_name'].to_numpy(),
        features['object area avg'].to_numpy(),
        features['total area avg'].to_numpy())

    mask = features['Video_name'].isin(feature_controled_video_groups)
    plt.scatter(features['velocity avg'],
                features['percent_of_frame_w_object'], label='all')
    plt.scatter(features[mask]['velocity avg'],
                features[mask]['percent_of_frame_w_object'],
                label='feature controlled')
    plt.xlabel('object velocity')
    plt.ylabel('percent of frames with object')
    plt.ylim(0, 1.1)
    plt.xlim(1, 3.5)

    # assign feature controlled videos into bins
    feature1_bins, feature2_bins, bin_videos = divide_bins(
        features[mask]['Video_name'].to_numpy(),
        features[mask]['velocity avg'].to_numpy(),
        features[mask]['percent_of_frame_w_object'].to_numpy(), NUM_BINS)
    plt.figure()
    all_bin_videos = []
    for key in bin_videos:
        all_bin_videos.extend(bin_videos[key])
    mask = features['Video_name'].isin(all_bin_videos)
    plt.scatter(features[mask]['velocity avg'],
                features[mask]['percent_of_frame_w_object'],
                label='feature controled videos')
    for val in feature1_bins:
        plt.axvline(x=val, c='k', linestyle='--')
    for val in feature2_bins:
        plt.axhline(y=val, c='k', linestyle='--')

    # randomly pick NUM_CHOICE videos from each bin
    selected_videos = []
    for key in bin_videos:
        if bin_videos[key]:
            selected_videos.extend(
                np.random.choice(bin_videos[key], NUM_CHOICE).tolist())
    mask = features['Video_name'].isin(selected_videos)
    plt.scatter(features[mask]['velocity avg'],
                features[mask]['percent_of_frame_w_object'], label='selected')
    plt.ylim(0, 1.1)
    plt.xlim(1, 3.5)
    plt.xlabel('object velocity')
    plt.ylabel('percent of frames with object')
    plt.legend()

    results = []
    for name in VIDEOS:
        results.append(pd.read_csv(
            '~/Projects/benchmarking/videostorm/test_coverage_results/videostorm_coverage_results_{}.csv'.format(name)))
        # vs_results.append(pd.read_csv(
        #     '~/Projects/benchmarking/videostorm/youtube_overfitting_30s/videostorm_overfitting_results_{}.csv'.format(name)))
    results.append(pd.read_csv(
        '~/Projects/benchmarking/videostorm/waymo_720p_e2e/videostorm_e2e_waymo.csv'))
    # vs_results.append(pd.read_csv(
    #     '~/Projects/benchmarking/videostorm/waymo_overfitting/videostorm_overfitting_waymo.csv'))

    results = pd.concat(results, ignore_index=True)
    plt.figure()
    mask = results['video_name'].isin(selected_videos)
    plt.scatter(results['gpu time'], results['f1'], label='all', s=1)
    plt.scatter(results[mask]['gpu time'],
                results[mask]['f1'], label='selected')
    plt.ylim(0, 1.1)
    plt.xlim(0, 1.1)
    plt.xlabel('gpu time')
    plt.ylabel('f1')
    plt.legend()

    kitti_results = pd.read_csv(
        '../videostorm/kitti_e2e/videostorm_e2e_kitti.csv')
    canda_results = pd.read_csv(
        '../videostorm/canada_e2e/videostorm_e2e_canada.csv')
    with open('../plot/vs_video_selection.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['coverage video', 'coverage gpu', 'coverage f1',
                         'selected video', 'selected gpu', 'selected f1',
                         'kitti video', 'kitti gpu', 'kitti_f1',
                         'long video', 'long video gpu', 'long video f1'])
        for values in zip_longest(
            results['video_name'], results['gpu time'], results['f1'],
            results[mask]['video_name'], results[mask]['gpu time'],
            results[mask]['f1'],
            kitti_results['video_name'], kitti_results['gpu time'],
            kitti_results['f1'],
            canda_results['video_name'], canda_results['gpu time'],
                canda_results['f1']):
            writer.writerow(values)

    plt.show()


def aws_selection():
    features = []
    features.append(pd.read_csv(
        'video_features_30s/allvideo_features_long_add_width_20_filter.csv'))
    features.append(pd.read_csv(
        'video_features_30s/waymovideo_features_long_add_width_20_filter.csv'))
    features = pd.concat(features, ignore_index=True)
    print(features.shape)
    features = features.dropna()
    print(features.shape)

    feature_controled_video_groups = control_features(
        features['Video_name'].to_numpy(),
        features['velocity avg'].to_numpy(),
        features['percent_of_frame_w_object'].to_numpy())
    # print(feature_controled_video_groups)

    # plt.scatter(features['Video_name'].isin(feature_controled_video_groups))
    mask = features['Video_name'].isin(feature_controled_video_groups)
    plt.scatter(features[mask]['object area avg'],
                features[mask]['total area avg'], s=3)
    plt.xlabel('average object size')
    plt.ylabel('average total object size')
    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)

    feature1_bins, feature2_bins, bin_videos = divide_bins(
        features[mask]['Video_name'].to_numpy(),
        features[mask]['object area avg'].to_numpy(),
        features[mask]['total area avg'].to_numpy(), NUM_BINS)
    plt.figure()
    all_bin_videos = []
    for key in bin_videos:
        all_bin_videos.extend(bin_videos[key])
    mask = features['Video_name'].isin(all_bin_videos)
    plt.scatter(features[mask]['object area avg'],
                features[mask]['total area avg'], s=3,
                label='feature controled videos')
    for val in feature1_bins:
        plt.axvline(x=val, c='k', linestyle='--')
    for val in feature2_bins:
        plt.axhline(y=val, c='k', linestyle='--')

    selected_videos = []
    for key in bin_videos:
        if bin_videos[key]:
            selected_videos.extend(
                np.random.choice(bin_videos[key], NUM_CHOICE).tolist())
    mask = features['Video_name'].isin(selected_videos)
    plt.scatter(features[mask]['object area avg'],
                features[mask]['total area avg'], s=3, label='selected')
    plt.xlabel('average object size')
    plt.ylabel('average total object size')
    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)
    plt.legend()

    results = []
    for name in VIDEOS:
        # aws_results.append(pd.read_csv(
        #     '~/Projects/benchmarking/awstream/overfitting_results_30s_30s_label_merge/awstream_spatial_overfitting_results_{}.csv'.format(name)))
        results.append(pd.read_csv(
            '~/Projects/benchmarking/awstream/e2e_results_30s_10s/awstream_e2e_results_{}.csv'.format(name)))
    # aws_results.append(pd.read_csv(
    #     '~/Projects/benchmarking/awstream/awstream_overfitting_waymo_30s.csv'))
    results.append(pd.read_csv(
        '~/Projects/benchmarking/awstream/awstream_e2e_waymo.csv'))

    results = pd.concat(results, ignore_index=True)
    plt.figure()
    mask = results['dataset'].isin(selected_videos)
    plt.scatter(results['bandwidth'], results['f1'], label='all', s=1)
    plt.scatter(results[mask]['bandwidth'],
                results[mask]['f1'], label='selected')
    plt.xlabel('bandwidth')
    plt.ylabel('f1')
    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)
    plt.legend()

    kitti_results = pd.read_csv('../awstream/awstream_e2e_kitti.csv')
    plt.figure()
    plt.scatter(results['bandwidth'], results['f1'], label='all', s=1)
    plt.scatter(kitti_results['bandwidth'], kitti_results['f1'], label='kitti')
    # canda_vs_results = pd.read_csv(
    #     '../videostorm/canada_e2e/videostorm_e2e_canada.csv')
    with open('../plot/aws_video_selection.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(
            ['coverage video', 'coverage bw', 'coverage f1',
             'selected video', 'selected bw', 'selected f1',
             'kitti video', 'kitti bw', 'kitti_f1',
             'long video', 'long video bw', 'long video f1'])
        for values in zip_longest(
            results['dataset'], results['bandwidth'], results['f1'],
            results[mask]['dataset'], results[mask]['bandwidth'],
            results[mask]['f1'],
            kitti_results['bandwidth'], kitti_results['f1'],
            results[results['dataset'].str.contains(
                'crossroad2')]['bandwidth'],
            results[results['dataset'].str.contains(
                'crossroad2')]['f1']):
            print(values)
            writer.writerow(values)

    plt.show()


if __name__ == '__main__':
    main()
# def no_selection_old():
#     videos = ['cropped_crossroad4', 'cropped_crossroad4_2',
#               'cropped_crossroad5', 'cropped_driving2']
#     features = []
#     features.append(pd.read_csv(
#         'video_features_30s/noscopevideo_features_long_add_width_20_filter.csv'))
#     features.append(pd.read_csv(
#         'video_features_30s/noscopevideo_features_long_add_width_20_filter.csv'))
#     features = pd.concat(features, ignore_index=True)
#     print(features.shape)
#     features = features.dropna()
#     print(features.shape)
#
#     feature_controled_video_groups = control_features(
#         features['Video_name'].to_numpy(),
#         features['velocity avg'].to_numpy(),
#         features['percent_of_frame_w_object'].to_numpy())
#     # print(feature_controled_video_groups)
#
#     # plt.scatter(features['Video_name'].isin(feature_controled_video_groups))
#     mask = features['Video_name'].isin(feature_controled_video_groups)
#     plt.scatter(features[mask]['nb_distinct_classes'],
#                 features[mask]['similarity'], s=3)
#     plt.xlabel('number of distinct classes')
#     plt.ylabel('similarity')
#     # plt.xlim(0, 1.1)
#     # plt.ylim(0, 1.1)
#
#     feature1_bins, feature2_bins, bin_videos = divide_bins(
#         features[mask]['Video_name'].to_numpy(),
#         features[mask]['nb_distinct_classes'].to_numpy(),
#         features[mask]['similarity'].to_numpy(), NUM_BINS)
#     plt.figure()
#     # pdb.set_trace()
#     all_bin_videos = []
#     for key in bin_videos:
#         all_bin_videos.extend(bin_videos[key])
#     mask = features['Video_name'].isin(all_bin_videos)
#     plt.scatter(features[mask]['nb_distinct_classes'],
#                 features[mask]['similarity'], s=3,
#                 label='feature controled videos')
#     for val in feature1_bins:
#         plt.axvline(x=val, c='k', linestyle='--')
#     for val in feature2_bins:
#         plt.axhline(y=val, c='k', linestyle='--')
#
#     selected_videos = []
#     for key in bin_videos:
#         if bin_videos[key]:
#             selected_videos.extend(
#                 np.random.choice(bin_videos[key], NUM_CHOICE).tolist())
#     mask = features['Video_name'].isin(selected_videos)
#     plt.scatter(features[mask]['nb_distinct_classes'],
#                 features[mask]['similarity'], s=3, label='selected')
#     plt.xlabel('number of distinct classes')
#     plt.ylabel('similarity')
#     # plt.xlim(0, 1.1)
#     # plt.ylim(0, 1.1)
#     plt.legend()
#
#     no_results = []
#     for name in videos:
#         # aws_results.append(pd.read_csv(
#         #     '~/Projects/benchmarking/awstream/overfitting_results_30s_30s_label_merge/awstream_spatial_overfitting_results_{}.csv'.format(name)))
#         no_results.append(pd.read_csv(
#             '~/Projects/benchmarking/noscope/results/noscope_result_{}.csv'.format(name)))
#     # aws_results.append(pd.read_csv(
#     #     '~/Projects/benchmarking/awstream/awstream_overfitting_waymo_30s.csv'))
#     # no_results.append(pd.read_csv(
#     #     '~/Projects/benchmarking/awstream/awstream_e2e_waymo.csv'))
#
#     no_results = pd.concat(no_results, ignore_index=True)
#     plt.figure()
#     mask = no_results['video_name'].isin(selected_videos)
#     plt.scatter(no_results['gpu'],
#                 no_results['f1'], label='all', s=1)
#     plt.scatter(no_results[mask]['gpu'],
#                 no_results[mask]['f1'], label='selected')
#     plt.xlabel('gpu')
#     plt.ylabel('f1')
#     plt.xlim(0, 1.1)
#     plt.ylim(0, 1.1)
#     plt.legend()
#
#     # kitti_aws_results = pd.read_csv(
#     #     '../awstream/awstream_e2e_kitti.csv')
#     # plt.figure()
#     # plt.scatter(aws_results['bandwidth'],
#     #             aws_results['f1'], label='all', s=1)
#     # plt.scatter(kitti_aws_results['bandwidth'],
#     #             kitti_aws_results['f1'], label='kitti')
#     # canda_vs_results = pd.read_csv(
#     #     '../videostorm/canada_e2e/videostorm_e2e_canada.csv')
#     with open('../plot/no_video_selection.csv', 'w') as f:
#         writer = csv.writer(f)
#         writer.writerow(['coverage perf', 'coverage acc',
#                          'selected perf', 'selected acc',
#                          'kitti perf', 'kitti_acc',
#                          'canada perf', 'canda acc'])
#         for values in zip_longest(no_results['gpu'],
#                                   no_results['f1'],
#                                   no_results[mask]['gpu'],
#                                   no_results[mask]['f1'],
#                                   # kitti_aws_results['bandwidth'],
#                                   # kitti_aws_results['f1'],
#                                   # canda_aws_results['bandwidth'],
#                                   # canda_aws_results['f1']
#                                   ):
#             writer.writerow(values)
#
#     plt.show()
