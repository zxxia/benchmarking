""" Feature Scanning Scirpt """
import argparse
# from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from utils.utils import load_metadata
from feature_analysis.helpers import load_video_features, nonzero, nonan, \
    load_videostorm_profile, sample_video_features


# DATASET = 't_crossroad'
# PATH = '/mnt/data/zhujun/dataset/Youtube/{}/1080p/profile/'.format(DATASET)
# PATH = '/mnt/data/zhujun/dataset/Youtube/{}/720p/profile/'.format(DATASET)
# METADATA_FILE = '/mnt/data/zhujun/dataset/Youtube/{}/metadata.json'.format(
#         DATASET)
# SAMPLE_EVERY_N_FRAME_LIST = [1, 10, 50, 100]
SAMPLE_EVERY_N_FRAME_LIST = [1, 10]
# PLOT_FLAG = False
PLOT_FLAG = True


def plot_cdf(data, num_bins, title, legend, xlabel):
    """ Use the histogram function to bin the data """
    counts, bin_edges = np.histogram(data, bins=num_bins, density=True)
    d_x = bin_edges[1] - bin_edges[0]
    # Now find the cdf
    cdf = np.cumsum(counts) * d_x
    # And finally plot the cdf
    plt.plot(bin_edges[1:], cdf, label=legend)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('CDF')
    plt.ylim([0, 1.1])
    plt.legend()


def parse_args():
    """ parse arguments """

    parser = argparse.ArgumentParser(description="Feature scanning script")
    parser.add_argument("--video", type=str, help="video name")
    parser.add_argument("--metadata", type=str, required=True,
                        help="metadata file in Json")
    parser.add_argument("--feature_file", type=str, required=True,
                        help="feature file")
    parser.add_argument("--feature_file_simple", type=str,
                        help="feature file of simple model")
    parser.add_argument("--output_file", type=str, help="output file")
    parser.add_argument("--short_video_length", type=int, required=True,
                        help="short video length in seconds")
    args = parser.parse_args()
    return args


def main():
    """ feature scanning """
    args = parse_args()
    # short_video_length = args.short_video_length

    metadata = load_metadata(args.metadata)
    frame_cnt = metadata['frame count']
    fps = metadata['frame rate']
    nb_short_videos = frame_cnt // (args.short_video_length * fps)
    # feature_name = args.feature
    # feature_name = 'Object Area'
    # feature_name = 'Total Object Area'
    feature_name = 'Object Velocity'

    video_features = load_video_features(args.feature_file)
    # filter out object smaller than a certain value
    for i in video_features:
        obj_area = video_features[i]['Object Area']
        filtered_obj_area = [val for val in obj_area if val > 0.02]
        video_features[i]['Object Area'] = filtered_obj_area

    if args.feature_file_simple:
        video_features_simple = load_video_features(args.feature_file_simple)
        for i in video_features_simple:
            obj_area = video_features_simple[i]['Object Area']
            filtered_obj_area = [val for val in obj_area if val > 0.02]
            video_features_simple[i]['Object Area'] = filtered_obj_area
    # f_out = open(args.output_file, 'a+', 1)

    for n_frame in SAMPLE_EVERY_N_FRAME_LIST:
        shvid_fts, _ = sample_video_features(video_features, metadata,
                                             args.short_video_length, n_frame)
        sampled_vals, sampled_medians, sampled_means, sampled_75_percent, \
            sampled_25_percent = process_short_video_features(shvid_fts,
                                                              nb_short_videos,
                                                              feature_name)
        if args.feature_file_simple:
            shvid_fts_simple, _ = \
                sample_video_features(video_features_simple, metadata,
                                      args.short_video_length, n_frame)
            sampled_vals_simple, sampled_medians_simple, sampled_means_simple,\
                sampled_75_percent_simple, sampled_25_percent_simple = \
                process_short_video_features(shvid_fts_simple, nb_short_videos,
                                             feature_name)

        if n_frame == 1:
            original_vals = sampled_vals
            original_medians = sampled_medians
            original_means = sampled_means
            original_75_percent = sampled_75_percent
            original_25_percent = sampled_25_percent
            if args.feature_file_simple:
                original_vals_simple = sampled_vals_simple
                original_medians_simple = sampled_medians_simple
                original_means_simple = sampled_means_simple
                original_75_percent_simple = sampled_75_percent_simple
                original_25_percent_simple = sampled_25_percent_simple
        # print(n_frame, len(short_vid_to_frame_id[key]))
        # sampled_medians = [x for x in sampled_medians if not np.isnan(x)]
        # print(len(sampled_vals))
        # print(sample_every_n_frame,
        #       len(short_vid_to_frame_id[key]),
        #       len(short_vid_to_velo[key])) #sampled_medians))

        if PLOT_FLAG:
            plt.figure(1)
            # plt.xlim([1, 6])
            plt.xlim([0, 0.1])
            plot_cdf(sampled_vals, 10000,
                     '{}: {} in 30s'.format(args.video, feature_name),
                     'every {} frames'.format(n_frame), feature_name)
            if args.feature_file_simple:
                plot_cdf(sampled_vals_simple, 10000,
                         '{} : {} in 30s'.format(args.video, feature_name),
                         'mobilenet every {} frames'.format(n_frame),
                         feature_name)
            # plt.savefig('{}: {} in 30s.png'.format(dataset, feature_name))
            plt.figure(2)
            # plt.xlim([1, 6])
            plt.xlim([0, 0.1])
            plot_cdf(nonan(sampled_medians.values()), 10000,
                     '{}: {} in 30s (median)'.format(args.video, feature_name),
                     'every {} frames'.format(n_frame), feature_name)
            if args.feature_file_simple:
                plot_cdf(nonan(sampled_medians_simple.values()), 10000,
                         '{}: {} in 30s (median)'
                         .format(args.video, feature_name),
                         'mobilenet every {} frames'
                         .format(n_frame), feature_name)
            # plt.savefig('{}: {} in 30s median.png'.format(dataset,
            #                                               feature_name))
            plt.figure(3)
            # plt.xlim([1, 6])
            plt.xlim([0, 0.1])
            plot_cdf(nonan(sampled_means.values()), 10000,
                     '{}: {} in 30s (mean)'.format(args.video, feature_name),
                     'every {} frames'.format(n_frame), feature_name)
            if args.feature_file_simple:
                plot_cdf(nonan(sampled_means_simple.values()), 10000,
                         '{}: {} in 30s (mean)'.format(args.video,
                                                       feature_name),
                         'mobilenet every {} frames'
                         .format(n_frame), feature_name)
            # plt.savefig('{}: {} in 30s mean.png'.format(dataset,
            # feature_name))
            plt.figure(4)
            # plt.xlim([1, 6])
            plt.xlim([0, 0.1])
            plot_cdf(nonan(sampled_75_percent.values()), 10000,
                     '{}: {} in 30s (75 percentile)'
                     .format(args.video, feature_name),
                     'every {} frames'.format(n_frame), feature_name)
            if args.feature_file_simple:
                plot_cdf(nonan(sampled_75_percent_simple.values()), 10000,
                         '{}: {} in 30s (75 percentile)'
                         .format(args.video, feature_name),
                         'mobilenet every {} frames'
                         .format(n_frame), feature_name)
            # plt.savefig('{}: {} in 30s 75.png'.format(dataset, feature_name))
            plt.figure(5)
            # plt.xlim([1, 6])
            plt.xlim([0, 0.1])
            plot_cdf(nonan(sampled_25_percent.values()), 10000,
                     '{}: {} in 30s (25 percentile)'
                     .format(args.video, feature_name),
                     'every {} frames'.format(n_frame), feature_name)
            if args.feature_file_simple:
                plot_cdf(nonan(sampled_25_percent_simple.values()), 10000,
                         '{}: {} in 30s (25 percentile)'
                         .format(args.video, feature_name),
                         'mobilenet every {} frames'
                         .format(n_frame), feature_name)
            # plt.savefig('{}: {} in 30s 25.png'.format(dataset, feature_name))
    # '/data/zxxia/benchmarking/videostorm/baseline_profile_10_15/videostorm_baseline_profile_{}.csv'.format(video))
    # '/data/zxxia/benchmarking/videostorm/baseline_profile_10_17_10s_50/videostorm_baseline_profile_{}.csv'.format(video))
    # '/data/zxxia/benchmarking/videostorm/baseline_profile_30s_100/videostorm_baseline_profile_{}.csv'.format(video))
    # baseline_file = '/data/zxxia/benchmarking/videostorm/baseline_profile_30s_100/videostorm_baseline_profile_{}.csv'.format(dataset)
    baseline_file = '/data/zxxia/benchmarking/videostorm/baseline_profile_30s_20/videostorm_baseline_profile_{}.csv'.format(args.video)
    profile_file = '/data/zxxia/benchmarking/videostorm/overfitting_profile_10_14/videostorm_overfitting_profile_{}.csv'.format(args.video)
    # profile_file = '/data/zxxia/benchmarking/videostorm/overfitting_profile_10_21_30s/videostorm_overfitting_profile_{}.csv'.format(dataset)
    original_good_percent, original_bad_percent, \
        scan_good_percent, scan_bad_percent, baseline_good_percent, \
        baseline_bad_percent = \
        compute_acc(original_means, sampled_means_simple, args.video,
                    profile_file, baseline_file)
        # compute_acc(original_means, sampled_means, dataset, baseline_file)
        # compute_acc(original_means, original_means_simple, dataset, baseline_file)
    # f_out.write(','.join([dataset, str(original_good_percent),
    #                       str(original_bad_percent),
    #                       str(scan_good_percent), str(scan_bad_percent),
    #                       str(baseline_good_percent),
    #                       str(baseline_bad_percent)])+'\n')

    plt.show()


def process_short_video_features(shvid_fts, nb_short_videos, feature_name):
    """ compute some statistics of short video features """

    sampled_vals = list()
    sampled_medians = dict()
    sampled_means = dict()
    sampled_75_percent = dict()
    sampled_25_percent = dict()
    print('num of short vid={}'.format(len(shvid_fts.keys())))
    for i in range(nb_short_videos):
        processed_fts = nonzero(shvid_fts[i][feature_name])

        if processed_fts:
            sampled_vals.extend(processed_fts)
            sampled_medians[i] = np.median(processed_fts)
            sampled_means[i] = np.mean(processed_fts)
            sampled_75_percent[i] = np.quantile(processed_fts, 0.75)
            sampled_25_percent[i] = np.quantile(processed_fts, 0.25)
        else:
            sampled_means[i] = np.nan
            sampled_medians[i] = np.nan
            sampled_75_percent[i] = np.nan
            sampled_25_percent[i] = np.nan
    return sampled_vals, sampled_medians, sampled_means, sampled_75_percent, \
        sampled_25_percent


def compute_acc(original_features, sampled_features, video, profile_file,
                baseline_file):
    """ measure feature scanning effectiveness """
    vids, perf_dict, acc_dict = load_videostorm_profile(profile_file)
    acc_1 = []
    acc_2 = []
    acc_4 = []
    acc_8 = []
    for key in vids:
        for perf, acc in zip(perf_dict[key], acc_dict[key]):
            if perf == 0.1:
                acc_1.append(acc)
            if perf == 0.2:
                acc_2.append(acc)
            if perf == 0.4:
                acc_4.append(acc)
            if 0.8 <= perf <= 0.9:
                acc_8.append(acc)

    # plt.plot(np.arange(len(vids)), acc_1, '.', c='k', label='fps=0.1')
    plt.plot(np.arange(len(vids)), acc_2, '.-', c='r', label='fps=0.2')
    # plt.plot(np.arange(len(vids)), acc_4, '.', c='b', label='fps=0.4')
    # plt.plot(np.arange(len(vids)), acc_8, '.', c='g', label='fps=0.833')
    plt.title(video)
    plt.legend()
    plt.ylim([0, 1])
    target_perf = 0.20
    acc_at_target_perf = []
    for key in vids:
        for perf, acc in zip(perf_dict[key], acc_dict[key]):
            if perf == target_perf:
                acc_at_target_perf.append(acc)
    nb_good = 0
    nb_bad = 0
    for acc in acc_at_target_perf:
        # if acc >= 0.80:
        if acc >= 0.85:
            nb_good += 1
        # if acc <= 0.75:
        if acc <= 0.50:
            nb_bad += 1
    original_good_percent = nb_good/len(acc_at_target_perf)
    original_bad_percent = nb_bad/len(acc_at_target_perf)
    print('num of original acc={}'.format(len(acc_at_target_perf)))
    print("original good percent={}, bad percent={}"
          .format(original_good_percent, original_bad_percent))

    nb_good = 0
    nb_bad = 0
    for i in sorted(sampled_features.keys()):
        # if sampled_features[i] <= 1.75:
        if sampled_features[i] <= 1.4:
            nb_good += 1
        if sampled_features[i] >= 2.5:
            nb_bad += 1
        # print(i, original_features[i], sampled_features[i])

    cnt = len([x for x in sampled_features if not np.isnan(x)])
    scan_good_percent = nb_good/cnt
    scan_bad_percent = nb_bad/cnt
    print('num of sampled features={}'.format(cnt))
    print("scanned good percent={}, bad percent={}"
          .format(scan_good_percent, scan_bad_percent))

    nb_good = 0
    nb_bad = 0

    for i in sorted(original_features.keys()):
        # if original_features[i] <= 1.75:
        if original_features[i] <= 1.4:
            nb_good += 1
        if original_features[i] >= 2.5:
            nb_bad += 1
        # print(original_features[i], sampled_features[i])

    cnt = len([x for x in original_features if not np.isnan(x)])
    # cnt = len(original_features)
    original_ft_good_percent = nb_good/cnt
    original_ft_bad_percent = nb_bad/cnt
    print('num of original features={}'.format(cnt))
    print("original good percent={}, bad percent={}"
          .format(original_ft_good_percent, original_ft_bad_percent))

    # acc_at_target_perf = []
    # vids, perf_dict, acc_dict = load_videostorm_profile(baseline_file)
    # for key in vids:
    #     for perf, acc in zip(perf_dict[key], acc_dict[key]):
    #         if perf == target_perf:
    #             acc_at_target_perf.append(acc)
    offset = 445
    nb_good = 0
    nb_bad = 0
    for acc in acc_at_target_perf[offset:offset+57]:
        if acc >= 0.85:
            nb_good += 1
        if acc <= 0.50:
            nb_bad += 1
    baseline_good_percent = nb_good/len(acc_at_target_perf[offset:offset+57])
    baseline_bad_percent = nb_bad/len(acc_at_target_perf[offset:offset+57])
    print('num of baseline acc={}'.format(len(acc_at_target_perf[offset:offset+57])))
    print('num of good cases={}'.format(nb_good))
    print("baseline good percent={}, bad percent={}"
          .format(baseline_good_percent, baseline_bad_percent))

    return original_good_percent, original_bad_percent, \
        scan_good_percent, scan_bad_percent, baseline_good_percent, \
        baseline_bad_percent


if __name__ == '__main__':
    main()
