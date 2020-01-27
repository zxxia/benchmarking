import csv
import os
import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, pearsonr
from sklearn.feature_selection import mutual_info_classif

VIDEOS = ['crossroad', 'crossroad2', 'crossroad3', 'crossroad4', 'drift',
          'driving1', 'driving_downtown', 'highway',
          'nyc', 'jp', 'lane_split',  # 'driving2',
          'motorway',  'park', ]  # 'russia',
# 'russia1', 'traffic', 'tw', 'tw1',
# 'tw_under_bridge']
# VIDEOS = ['highway', 'motorway', 'jp', 'drift', 'russia1']
FEATURE_ROOT = '~/Projects/benchmarking/feature_analysis/video_features_30s'


def write_gnu_data(filename, perfs_to_plot, avg_velo, percent, avg_obj_size,
                   nb_distinct_classes):
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['number', 'name', 'pearson coefficient',
                         'pearson p', 'kendalltau coefficient',
                         'kendalltau p'])
        writer.writerows([(1, 'avg velocity vs. gpu',
                           abs(pearsonr(avg_velo, perfs_to_plot)[0]),
                           pearsonr(avg_velo, perfs_to_plot)[1],
                           abs(kendalltau(avg_velo, perfs_to_plot)[0]),
                           kendalltau(avg_velo, perfs_to_plot)[1]),
                          (2, 'percentage of frame with object vs. gpu',
                              abs(pearsonr(percent, perfs_to_plot)[0]),
                              pearsonr(percent, perfs_to_plot)[1],
                              abs(kendalltau(percent, perfs_to_plot)[0]),
                              kendalltau(percent, perfs_to_plot)[1]),
                          (3, 'avg object size vs. gpu',
                              abs(pearsonr(avg_obj_size, perfs_to_plot)[0]),
                              pearsonr(avg_obj_size, perfs_to_plot)[1],
                              abs(kendalltau(avg_obj_size, perfs_to_plot)[0]),
                              kendalltau(avg_obj_size, perfs_to_plot)[1]),
                          (4, 'nb distinct clases vs. gpu',
                              abs(pearsonr(nb_distinct_classes,
                                           perfs_to_plot)[0]),
                              pearsonr(nb_distinct_classes, perfs_to_plot)[1],
                              abs(kendalltau(nb_distinct_classes,
                                             perfs_to_plot)[0]),
                              kendalltau(nb_distinct_classes,
                                         perfs_to_plot)[1]),
                          ])


# VIDEOS = ['driving2', 'park', 'driving1']


def vs_feature_correlation():
    features = pd.read_csv(os.path.join(
        FEATURE_ROOT, 'allvideo_features_long_add_width_20_filter.csv'))
    vs_path = '/home/zxxia/Projects/benchmarking/videostorm/test_coverage_results/' \
        'videostorm_coverage_results_{}.csv'
    avg_velo = []
    avg_obj_size = []
    avg_tot_obj_size = []
    percent = []
    nb_distinct_classes = []
    videos_to_plot = []
    perfs_to_plot = []
    accs_to_plot = []
    for name in VIDEOS:
        vs_results = pd.read_csv(vs_path.format(name))
        mask = vs_results['f1'].between(0.9, 0.95)
        selected_videos = vs_results['video_name'].loc[mask].to_list()
        videos_to_plot.extend(vs_results['video_name'].loc[mask])
        perfs_to_plot.extend(vs_results['gpu time'].loc[mask].to_list())
        accs_to_plot.extend(vs_results['f1'].loc[mask].to_list())

        # features = pd.read_csv(
        #     os.path.join(FEATURE_ROOT, '{}_features.csv'.format(name)))
        # avg_velo.extend(features['average velocity'].loc[mask].to_list())
        # avg_obj_size.extend(
        #     features['average object size'].loc[mask].to_list())
        # avg_tot_obj_size.extend(
        #     features['average total object size'].loc[mask].to_list())
        # percent.extend(
        #     features['percentage of frame with object'].loc[mask].to_list())
        # nb_distinct_classes.extend(
        #     features['number of distinct classes'].loc[mask].to_list())
        mask = features['Video_name'].isin(selected_videos)
        avg_velo.extend(
            features['velocity avg'].loc[mask].to_list())
        avg_obj_size.extend(
            features['object area percentile10'].loc[mask].to_list())
        avg_tot_obj_size.extend(
            features['total area avg'].loc[mask].to_list())
        percent.extend(
            features['percent_of_frame_w_object'].loc[mask].to_list())
        nb_distinct_classes.extend(
            features['nb_distinct_classes'].loc[mask].to_list())
    assert len(avg_velo) == len(perfs_to_plot)
    write_gnu_data('vs_correlation.csv', perfs_to_plot, avg_velo, percent,
                   avg_obj_size, nb_distinct_classes)
    # plt.scatter(avg_velo, perfs_to_plot)
    # ax = plt.gca()
    # for perf, acc, vid in zip(avg_velo, perfs_to_plot,  videos_to_plot):
    #     # if 'drift' in vid:
    #     ax.annotate(vid, (perf, acc))
    # plt.ylim([0, 1])
    print('avg velo vs. bw kendall', kendalltau(avg_velo, perfs_to_plot))
    print('avg obj size vs. bw kendall',
          kendalltau(avg_obj_size, perfs_to_plot))
    print('avg tot obj size vs. bw kendall', kendalltau(
        avg_tot_obj_size, perfs_to_plot))
    print('percent vs. bw kendall', kendalltau(percent, perfs_to_plot))

    print('avg velo vs. bw pearson', pearsonr(avg_velo, perfs_to_plot))
    print('avg obj size vs. bw pearson',
          pearsonr(avg_obj_size, perfs_to_plot))
    print('avg tot obj size vs. bw pearson', pearsonr(
        avg_tot_obj_size, perfs_to_plot))
    print('percent vs. bw pearson', pearsonr(percent, perfs_to_plot))


def aws_feature_correlation():
    # aws_path = '/home/zxxia/Projects/benchmarking/plot/e2e_results_30s_10s/awstream_e2e_results_{}.csv'
    features = pd.read_csv(os.path.join(
        FEATURE_ROOT, 'allvideo_features_long_add_width_20_filter.csv'))
    aws_path = '/home/zxxia/Projects/benchmarking/awstream/e2e_results_label_merge/awstream_e2e_results_{}.csv'
    videos_to_plot = []
    perfs_to_plot = []
    accs_to_plot = []

    avg_velo = []
    avg_obj_size = []
    avg_tot_obj_size = []
    percent = []
    nb_distinct_classes = []
    for name in VIDEOS:
        aws_results = pd.read_csv(aws_path.format(name))
        mask = aws_results['f1'].between(0.9, 0.95)
        selected_videos = aws_results['dataset'].loc[mask].to_list()
        # pdb.set_trace()
        videos_to_plot.extend(
            aws_results['dataset'].loc[mask])
        perfs_to_plot.extend(
            aws_results['bandwidth'].loc[mask].to_list())
        accs_to_plot.extend(
            aws_results['f1'].loc[mask].to_list())
        # features = pd.read_csv(
        #     os.path.join(FEATURE_ROOT, '{}_features.csv'.format(name)))
        # avg_velo.extend(
        #     features['average velocity'].loc[mask].to_list())
        # avg_obj_size.extend(
        #     features['average object size'].loc[mask].to_list())
        # avg_tot_obj_size.extend(
        #     features['average total object size'].loc[mask].to_list())
        # percent.extend(
        #     features['percentage of frame with object'].loc[mask].to_list())
        # nb_distinct_classes.extend(
        #     features['number of distinct classes'].loc[mask].to_list())
        mask = features['Video_name'].isin(selected_videos)
        avg_velo.extend(
            features['velocity avg'].loc[mask].to_list())
        avg_obj_size.extend(
            features['object area percentile10'].loc[mask].to_list())
        avg_tot_obj_size.extend(
            features['total area avg'].loc[mask].to_list())
        percent.extend(
            features['percent_of_frame_w_object'].loc[mask].to_list())
        nb_distinct_classes.extend(
            features['nb_distinct_classes'].loc[mask].to_list())
    # add waymo results
    # aws_results = pd.read_csv(
    #     '~/Projects/benchmarking/awstream/awstream_e2e_waymo.csv')
    # mask = aws_results['f1'].between(0.9, 0.95)
    # selected_videos = aws_results['dataset'].loc[mask].to_list()
    # videos_to_plot.extend(
    #     aws_results['dataset'].loc[mask])
    # perfs_to_plot.extend(
    #     aws_results['bandwidth'].loc[mask].to_list())
    # accs_to_plot.extend(
    #     aws_results['f1'].loc[mask].to_list())
    # features = pd.read_csv(os.path.join(
    # FEATURE_ROOT, 'waymovideo_features_long_add_width_20_filter.csv'))
    # mask = features['Video_name'].isin(selected_videos)
    # avg_velo.extend(
    # features['velocity avg'].loc[mask].to_list())
    # avg_obj_size.extend(
    # features['object area percentile10'].loc[mask].to_list())
    # avg_tot_obj_size.extend(
    # features['total area avg'].loc[mask].to_list())
    # percent.extend(
    # features['percent_of_frame_w_object'].loc[mask].to_list())
    # nb_distinct_classes.extend(
    # features['nb_distinct_classes'].loc[mask].to_list())
    # avg_velo.extend(
    #     features['average velocity'].loc[mask].to_list())
    # avg_obj_size.extend(
    #     features[''].loc[mask].to_list())
    # avg_tot_obj_size.extend(
    #     features['average total object size'].loc[mask].to_list())
    # percent.extend(
    #     features['percentage of frame with object'].loc[mask].to_list())
    # nb_distinct_classes.extend(
    #     features['number of distinct classes'].loc[mask].to_list())
    # print(avg_velo)
    assert len(avg_velo) == len(perfs_to_plot)
    write_gnu_data('aws_correlation.csv', perfs_to_plot, avg_velo, percent,
                   avg_obj_size, nb_distinct_classes)
    print('avg velo vs. bw kendall', kendalltau(avg_velo, perfs_to_plot))
    print('avg obj size vs. bw kendall',
          kendalltau(avg_obj_size, perfs_to_plot))
    print('avg tot obj size vs. bw kendall', kendalltau(
        avg_tot_obj_size, perfs_to_plot))
    print('percent vs. bw kendall', kendalltau(percent, perfs_to_plot))

    print('avg velo vs. bw pearson', pearsonr(avg_velo, perfs_to_plot))
    print('avg obj size vs. bw pearson',
          pearsonr(avg_obj_size, perfs_to_plot))
    print('avg tot obj size vs. bw pearson', pearsonr(
        avg_tot_obj_size, perfs_to_plot))
    print('percent vs. bw pearson', pearsonr(percent, perfs_to_plot))

    # print(pearsonr(avg_obj_size, perfs_to_plot))
    # features = np.hstack((np.array(avg_velo).reshape(-1, 1),
    #                       np.array(percent).reshape(-1, 1),
    #                       np.array(avg_obj_size).reshape(-1, 1),
    #                       np.array(nb_distinct_classes).reshape(-1, 1)))

    # pdb.set_trace()
    # print(mutual_info_classif(features, perfs_to_plot))

    # plt.scatter(avg_velo, perfs_to_plot, label='avg velocity')
    # plt.scatter(avg_obj_size, perfs_to_plot, label='avg obj size')
    # plt.legend()
    # plt.show()


def gl_feature_correlation():
    features = pd.read_csv(os.path.join(
        FEATURE_ROOT, 'allvideo_features_long_add_width_20_filter.csv'))
    # gl_path = '/data/zxxia/benchmarking/glimpse/e2e_results/glimpse_tracking_{}.csv'
    gl_path = '/data/zxxia/benchmarking/glimpse/e2e_results_kcf/glimpse_result_{}.csv'
    # gl_path = '/home/zxxia/Projects/benchmarking/glimpse/glimpse_e2e_tracking_results/glimpse_tracking_{}.csv'
    # gl_path = '/home/zxxia/Projects/benchmarking/glimpse/glimpse_e2e_perfect_tracking_results/glimpse_perfect_tracking_{}.csv'
    # gl_path = '/home/zxxia/Projects/benchmarking/glimpse/glimpse_e2e_frame_select_results/glimpse_frame_select_{}.csv'
    videos_to_plot = []
    perfs_to_plot = []
    accs_to_plot = []

    avg_velo = []
    avg_obj_size = []
    avg_tot_obj_size = []
    percent = []
    nb_distinct_classes = []
    for name in VIDEOS:
        gl_results = pd.read_csv(gl_path.format(name))
        mask = gl_results['f1'].between(0.9, 0.95)
        selected_videos = gl_results['video chunk'].loc[mask].to_list()
        videos_to_plot.extend(
            gl_results['video chunk'].loc[mask])
        perfs_to_plot.extend(
            gl_results['frame rate'].loc[mask].to_list())
        accs_to_plot.extend(
            gl_results['f1'].loc[mask].to_list())
        # features = pd.read_csv(
        #     os.path.join(FEATURE_ROOT, '{}_features.csv'.format(name)))
        # avg_velo.extend(
        #     features['average velocity'].loc[mask].to_list())
        # avg_obj_size.extend(
        #     features['average object size'].loc[mask].to_list())
        # avg_tot_obj_size.extend(
        #     features['average total object size'].loc[mask].to_list())
        # percent.extend(
        #     features['percentage of frame with object'].loc[mask].to_list())
        # nb_distinct_classes.extend(
        #     features['number of distinct classes'].loc[mask].to_list())
        mask = features['Video_name'].isin(selected_videos)
        avg_velo.extend(
            features['velocity avg'].loc[mask].to_list())
        avg_obj_size.extend(
            features['object area percentile10'].loc[mask].to_list())
        avg_tot_obj_size.extend(
            features['total area avg'].loc[mask].to_list())
        percent.extend(
            features['percent_of_frame_w_object'].loc[mask].to_list())
        nb_distinct_classes.extend(
            features['nb_distinct_classes'].loc[mask].to_list())
    write_gnu_data('gl_correlation.csv', perfs_to_plot, avg_velo, percent,
                   avg_obj_size, nb_distinct_classes)
    print('avg velo vs. bw kendall', kendalltau(avg_velo, perfs_to_plot))
    print('avg obj size vs. bw kendall',
          kendalltau(avg_obj_size, perfs_to_plot))
    print('avg tot obj size vs. bw kendall', kendalltau(
        avg_tot_obj_size, perfs_to_plot))
    print('percent vs. bw kendall', kendalltau(percent, perfs_to_plot))

    print('avg velo vs. bw pearson', pearsonr(avg_velo, perfs_to_plot))
    print('avg obj size vs. bw pearson',
          pearsonr(avg_obj_size, perfs_to_plot))
    print('avg tot obj size vs. bw pearson', pearsonr(
        avg_tot_obj_size, perfs_to_plot))
    print('percent vs. bw pearson', pearsonr(percent, perfs_to_plot))
    # pdb.set_trace()


def no_feature_correlation():
    no_path = '/home/zxxia/Projects/benchmarking/noscope/results/noscope_result_{}.csv'
    # Plot noscope results
    VIDEOS = ['cropped_crossroad4',  'cropped_crossroad4_2',
              'cropped_crossroad5', 'cropped_driving2']
    videos_to_plot = []
    perfs_to_plot = []
    accs_to_plot = []

    avg_velo = []
    avg_obj_size = []
    avg_tot_obj_size = []
    percent = []
    nb_distinct_classes = []
    for name in VIDEOS:
        filename = no_path.format(name)
        no_results = pd.read_csv(filename)
        features = pd.read_csv(
            os.path.join(FEATURE_ROOT, '{}_features.csv'.format(name)))
        shorter_len = min(features.shape[0], no_results.shape[0])
        features = features.iloc[0:shorter_len]
        no_results = no_results.iloc[0:shorter_len]
        print(no_results.shape, features.shape)

        mask = no_results['f1'].between(0.9, 0.95)
        # print(mask.shape)
        videos_to_plot.extend(
            no_results['video_name'].loc[mask].to_list())
        perfs_to_plot.extend(
            no_results['gpu'].loc[mask].to_list())
        accs_to_plot.extend(
            no_results['f1'].loc[mask].to_list())
        # print(no_results)
        # print(features)
        assert no_results.shape[0] == features.shape[0], print(
            no_results.shape, features.shape)
        avg_velo.extend(features['average velocity'].loc[mask].to_list())
        avg_obj_size.extend(
            features['average object size'].loc[mask].to_list())
        avg_tot_obj_size.extend(
            features['average total object size'].loc[mask].to_list())
        percent.extend(
            features['percentage of frame with object'].loc[mask].to_list())
        nb_distinct_classes.extend(
            features['number of distinct classes'].loc[mask].to_list())
    assert len(avg_velo) == len(perfs_to_plot)
    write_gnu_data('no_correlation.csv', perfs_to_plot, avg_velo, percent,
                   avg_obj_size, nb_distinct_classes)
    print('avg velo vs. bw kendall', kendalltau(avg_velo, perfs_to_plot))
    print('avg obj size vs. bw kendall',
          kendalltau(avg_obj_size, perfs_to_plot))
    print('avg tot obj size vs. bw kendall', kendalltau(
        avg_tot_obj_size, perfs_to_plot))
    print('percent vs. bw kendall', kendalltau(percent, perfs_to_plot))

    print('avg velo vs. bw pearson', pearsonr(avg_velo, perfs_to_plot))
    print('avg obj size vs. bw pearson',
          pearsonr(avg_obj_size, perfs_to_plot))
    print('avg tot obj size vs. bw pearson', pearsonr(
        avg_tot_obj_size, perfs_to_plot))
    print('percent vs. bw pearson', pearsonr(percent, perfs_to_plot))


def main():
    vs_feature_correlation()
    aws_feature_correlation()
    gl_feature_correlation()


if __name__ == '__main__':
    main()
