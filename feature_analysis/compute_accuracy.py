""" Compute feature scanning performance boost """
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import seaborn as sns
from utils.utils import load_metadata
from feature_analysis.helpers import load_videostorm_profile, nonzero, \
     load_video_features, sample_video_features
DATASET_LIST = sorted(['crossroad', 'crossroad2', 'crossroad3', 'crossroad4',
                       'drift', 'driving1', 'driving2',  # 'driving_downtown',
                       'highway', 'jp',  # 'jp_hw','highway_normal_traffic',
                       'lane_split', 'motorway', 'nyc', 'park', 'russia',
                       'russia1', 'traffic', 'tw', 'tw1', 'tw_road',
                       'tw_under_bridge', ])
# DATASET_LIST = ['road_trip']

SAMPLE_EVERY_N_FRAME_LIST = [1]

short_video_length = 30


def main():
    """ plot 3d figure """
    # load features from all video feature files
    acc_at_target_perf = []
    feat_to_plot = []
    mobilenet_feat_to_plot = []
    annot_to_plot = []
    plt.figure(0)
    axs = plt.axes(projection='3d')
    axs.set_xlabel('Object Velocity Mean')
    axs.set_ylabel('gpu processing time')
    axs.set_zlabel('accuracy')
    temporal_sampling_list = [20, 15, 10, 5, 4, 3, 2.5, 2, 1.8, 1.5, 1.2, 1]
    cmap = sns.cubehelix_palette(as_cmap=True)

    for video in DATASET_LIST:
        print(video)
        video_features = load_video_features('/data/zxxia/benchmarking/results/videos/{}/720p/profile/Video_features_{}_object_width_height_type_filter.csv'.format(video, video))
        mobilenet_video_features = load_video_features('/data/zxxia/benchmarking/results/videos/{}/720p/profile/Video_features_{}_object_width_height_type_filter_mobilenet.csv'.format(video, video))
        metadata = load_metadata('/data/zxxia/videos/{}/metadata.json'.format(video))
        frame_cnt = metadata['frame count']
        fps = metadata['frame rate']
        nb_short_videos = frame_cnt // (short_video_length * fps)
        feature_name = 'Object Velocity'
        # feature_name = 'Object Area'
        for n_frame in SAMPLE_EVERY_N_FRAME_LIST:
            shvid_fts, _ = sample_video_features(video_features, metadata,
                                                 short_video_length, n_frame)
            sampled_vals = list()
            sampled_medians = dict()
            sampled_means = dict()
            sampled_75_percent = dict()
            sampled_25_percent = dict()
            sampled_medians = dict()
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
            # use mobilenet detections to compute feature
            shvid_fts, _ = sample_video_features(mobilenet_video_features,
                                                 metadata, short_video_length,
                                                 n_frame)
            mobilenet_sampled_vals = list()
            mobilenet_sampled_medians = dict()
            mobilenet_sampled_means = dict()
            mobilenet_sampled_75_percent = dict()
            mobilenet_sampled_25_percent = dict()
            mobilenet_sampled_medians = dict()
            print('num of short vid={}'.format(len(shvid_fts.keys())))
            for i in range(nb_short_videos):
                processed_fts = nonzero(shvid_fts[i][feature_name])

                if processed_fts:
                    mobilenet_sampled_vals.extend(processed_fts)
                    mobilenet_sampled_medians[i] = np.median(processed_fts)
                    mobilenet_sampled_means[i] = np.mean(processed_fts)
                    mobilenet_sampled_75_percent[i] = np.quantile(
                        processed_fts, 0.75)
                    mobilenet_sampled_25_percent[i] = np.quantile(
                        processed_fts, 0.25)
                else:
                    mobilenet_sampled_means[i] = np.nan
                    mobilenet_sampled_medians[i] = np.nan
                    mobilenet_sampled_75_percent[i] = np.nan
                    mobilenet_sampled_25_percent[i] = np.nan

        _, perf_dict, acc_dict = load_videostorm_profile('/data/zxxia/benchmarking/videostorm/overfitting_profile_10_14/videostorm_overfitting_profile_{}.csv'.format(video))
        # _, perf_dict, acc_dict = load_videostorm_profile('/data/zxxia/benchmarking/videostorm/overfitting_profile_10_17_10s/videostorm_overfitting_profile_{}.csv'.format(video))
        assert len(perf_dict) == len(acc_dict)
        assert len(perf_dict) == len(sampled_means), "{} != {}".format(len(perf_dict), len(sampled_means))

        # plot 3D figure to indicate f1 vs feature at different fps
        plt.figure(0)
        for i in range(nb_short_videos):
            clip = video + '_' + str(i)
            axs.scatter(np.ones(len(perf_dict[clip]))*sampled_means[i],
                        perf_dict[clip], acc_dict[clip], s=5,
                        c=temporal_sampling_list, cmap=cmap)

        target_perf = 0.20
        for i in range(nb_short_videos):
            clip = video + '_' + str(i)
            for pf, ac in zip(perf_dict[clip], acc_dict[clip]):
                if pf == target_perf:
                    # print(pf, ac, sampled_means[i])
                    # plt.plot(sampled_means[i], ac)
                    acc_at_target_perf.append(ac)
                    feat_to_plot.append(sampled_means[i])
                    mobilenet_feat_to_plot.append(mobilenet_sampled_means[i])
                    annot_to_plot.append(clip)
                    # print(feat, pf, ac)
    plt.figure(1)
    plt.scatter(feat_to_plot, acc_at_target_perf, s=7, label='frcnn')
    plt.scatter(mobilenet_feat_to_plot, acc_at_target_perf, s=7,
                label='mobilenet')
    plt.legend()
    plt.ylim([0, 1])
    # plt.xlim([1, 6])
    for annot, feat, mobilenet_feat, acc in zip(annot_to_plot, feat_to_plot,
                                                mobilenet_feat_to_plot,
                                                acc_at_target_perf):
        if 'road_trip_499' in annot:
            plt.annotate(annot, (feat, acc))
            plt.annotate(annot, (mobilenet_feat, acc))
    plt.title("gpu proccessing time={}".format(target_perf))
    plt.xlabel('Object Velocity Mean')
    plt.ylabel('accuracy')
    plt.axhline(y=0.85, c='k', linestyle='--')
    plt.axhline(y=0.50, c='k', linestyle='--')
    plt.axvline(x=1.40, c='k', linestyle='--')
    plt.axvline(x=2.5, c='k', linestyle='--')
    plt.show()


if __name__ == '__main__':
    main()
