""" Compute feature scanning performance boost """
import os
import pdb
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import seaborn as sns
from utils.utils import load_metadata
from utils.model_utils import load_full_model_detection, \
    filter_video_detections, remove_overlappings, compute_area
from feature_analysis.helpers import load_awstream_profile, nonzero, \
     load_video_features, sample_video_features
from constants import COCOLabels, CAMERA_TYPES, RESOL_DICT

DATASET_LIST = sorted(['crossroad', 'crossroad2', 'crossroad3', 'crossroad4',
                       'drift', 'driving1', 'driving_downtown',
                       'highway', 'jp', 'driving2',
                       # 'jp_hw','highway_normal_traffic',
                       'lane_split', 'motorway', 'nyc', 'park',  # 'russia',
                       'russia1', 'traffic', 'tw', 'tw1',
                       # 'tw_road', 'tw_under_bridge',
                       ])

# DATASET_LIST = ['jp', 'highway', 'motorway', 'driving_downtown', 'nyc']

VIDEO_TO_DELETE = ['crossroad', 'nyc', 'russia', 'crossroad2',
                   'driving_downtown',
                   'tw_road', 'tw_under_bridge', 'tw1', 'tw', 'crossroad3']

SAMPLE_EVERY_N_FRAME_LIST = [1]

short_video_length = 30

RESULTS_PATH = '/data/zxxia/benchmarking/results/videos'

filterd_videos = ['crossroad4_0', 'crossroad4_1', 'crossroad4_10',
'crossroad4_11', 'crossroad4_12', 'crossroad4_13', 'crossroad4_14',
'crossroad4_15', 'crossroad4_16', 'crossroad4_17', 'crossroad4_18',
'crossroad4_19', 'crossroad4_2', 'crossroad4_20', 'crossroad4_21',
'crossroad4_22', 'crossroad4_23', 'crossroad4_24', 'crossroad4_25', 'crossroad4_26', 'crossroad4_27', 'crossroad4_28',
'crossroad4_29', 'crossroad4_3', 'crossroad4_30', 'crossroad4_31', 'crossroad4_32', 'crossroad4_33',
'crossroad4_34', 'crossroad4_35', 'crossroad4_36', 'crossroad4_37', 'crossroad4_38', 'crossroad4_39',
'crossroad4_4', 'crossroad4_5', 'crossroad4_6', 'crossroad4_7', 'crossroad4_8', 'crossroad4_9',
'crossroad_0', 'crossroad_1', 'crossroad_10', 'crossroad_11', 'crossroad_14', 'crossroad_15',
'crossroad_16', 'crossroad_17', 'crossroad_18', 'crossroad_19', 'crossroad_2', 'crossroad_20',
'crossroad_21', 'crossroad_23', 'crossroad_24', 'crossroad_25', 'crossroad_26', 'crossroad_27',
'crossroad_28', 'crossroad_29', 'crossroad_3', 'crossroad_30', 'crossroad_31', 'crossroad_32',
'crossroad_33', 'crossroad_34', 'crossroad_35', 'crossroad_36', 'crossroad_37', 'crossroad_38',
'crossroad_39', 'crossroad_40', 'crossroad_41', 'crossroad_42', 'crossroad_44', 'crossroad_45',
'crossroad_46', 'crossroad_47', 'crossroad_48', 'crossroad_49', 'crossroad_50', 'crossroad_51',
'crossroad_52', 'crossroad_55', 'crossroad_56', 'crossroad_57', 'crossroad_58', 'crossroad_59',
'crossroad_6', 'crossroad_60', 'crossroad_61', 'crossroad_62', 'crossroad_63', 'crossroad_7',
'crossroad_8', 'crossroad_9', 'driving2_17', 'driving2_18', 'driving2_19', 'driving2_20',
'driving2_26', 'highway_0', 'highway_1', 'highway_10', 'highway_11', 'highway_12', 'highway_13',
'highway_14', 'highway_15', 'highway_16', 'highway_17', 'highway_18', 'highway_19', 'highway_2',
'highway_20', 'highway_21', 'highway_22', 'highway_23', 'highway_24', 'highway_25', 'highway_26',
'highway_27', 'highway_28', 'highway_29', 'highway_3', 'highway_30', 'highway_31', 'highway_32',
'highway_33', 'highway_34', 'highway_35', 'highway_36', 'highway_37', 'highway_38', 'highway_39',
'highway_4', 'highway_40', 'highway_41', 'highway_42', 'highway_43', 'highway_44', 'highway_45',
'highway_46', 'highway_47', 'highway_48', 'highway_49', 'highway_5', 'highway_50', 'highway_51',
'highway_52', 'highway_53', 'highway_54', 'highway_55', 'highway_56', 'highway_57', 'highway_58',
'highway_59', 'highway_6', 'highway_60', 'highway_61', 'highway_62', 'highway_63', 'highway_64',
'highway_65', 'highway_66', 'highway_67', 'highway_7', 'highway_8', 'highway_9', 'jp_0', 'jp_15',
'jp_16', 'jp_26', 'jp_29', 'jp_30', 'jp_31', 'jp_33', 'jp_38', 'jp_39', 'jp_4', 'lane_split_0',
'lane_split_1', 'lane_split_10', 'lane_split_11', 'lane_split_12', 'lane_split_13', 'lane_split_14',
'lane_split_15', 'lane_split_16', 'lane_split_3', 'lane_split_4', 'lane_split_5', 'lane_split_6',
'lane_split_7', 'lane_split_8', 'lane_split_9', 'motorway_0', 'motorway_1', 'motorway_10',
'motorway_11', 'motorway_12', 'motorway_13', 'motorway_14', 'motorway_15', 'motorway_16',
'motorway_17', 'motorway_18', 'motorway_2', 'motorway_3', 'motorway_4', 'motorway_5', 'motorway_6',
'motorway_7', 'motorway_8', 'motorway_9', 'park_32', 'park_33', 'park_34', 'russia1_12', 'russia1_20',
'russia1_23', 'russia1_28', 'russia1_3', 'russia1_31', 'russia1_32', 'russia1_33', 'traffic_0',
'traffic_1']


def load_detections(video, dt_file, resol):
    """ load and filter  """
    dts, nb_frame = load_full_model_detection(dt_file)
    if video in CAMERA_TYPES['moving']:
        dts = filter_video_detections(dts,
                                      target_types={COCOLabels.CAR.value,
                                                    COCOLabels.BUS.value,
                                                    COCOLabels.TRAIN.value,
                                                    COCOLabels.TRUCK.value},
                                      height_range=(RESOL_DICT[resol][1]//20,
                                                    RESOL_DICT[resol][1]))
    else:
        dts = filter_video_detections(dts,
                                      target_types={COCOLabels.CAR.value,
                                                    COCOLabels.BUS.value,
                                                    COCOLabels.TRAIN.value,
                                                    COCOLabels.TRUCK.value},
                                      width_range=(0, RESOL_DICT[resol][0]/2),
                                      height_range=(RESOL_DICT[resol][0]//20,
                                                    RESOL_DICT[resol][0]/2))
    for frame_idx, bboxes in dts.items():
        # merge all vehicle labels into CAR
        for box_pos, box in enumerate(bboxes):
            box[4] = COCOLabels.CAR.value
            bboxes[box_pos] = box
        dts[frame_idx] = bboxes
        # remove overlappings to mitigate occultation
        dts[frame_idx] = remove_overlappings(bboxes, 0.3)
    return dts, nb_frame


def compute_video_features(dts, resol):
    video_features = {}
    for frame_idx, boxes in dts.items():
        areas = [compute_area(box)/(RESOL_DICT[resol][0]*RESOL_DICT[resol][1])
                 for box in boxes]
        features = {'Object Area': areas,
                    'Object Velocity': [],
                    'Total Object Area': np.sum(areas),
                    'Object Count': len(areas)}
        video_features[frame_idx] = features
    return video_features


def main():
    """ plot 3d figure """
    # load features from all video feature files
    acc_at_target_resol = []
    feat_to_plot = []
    # mobilenet_feat_to_plot = []
    annot_to_plot = []
    plt.figure(0)
    axs = plt.axes(projection='3d')
    axs.set_xlabel('Object Size Mean')
    axs.set_ylabel('Resolution')
    axs.set_zlabel('accuracy')
    resolution_list = [360, 480, 540, 720]
    cmap = sns.cubehelix_palette(as_cmap=True)

    for video in DATASET_LIST:
        # if video in VIDEO_TO_DELETE:
        #     continue
        # print(video)
        # # load video features
        resol = '720p'
        gt_file = os.path.join(RESULTS_PATH, video, resol,
                               'profile/updated_gt_FasterRCNN_COCO_no_filter.csv')
        dts, nb_frame = load_detections(video, gt_file, resol)
        video_features = compute_video_features(dts, resol)

        # video_features = load_video_features('/data/zxxia/benchmarking/results/videos/{}/720p/profile/Video_features_{}_object_width_height_type_filter.csv'.format(video, video))
        # mobilenet_video_features = load_video_features('/data/zxxia/benchmarking/results/videos/{}/720p/profile/Video_features_{}_object_width_height_type_filter_mobilenet.csv'.format(video, video))
        metadata = load_metadata('/data/zxxia/videos/{}/metadata.json'.format(video))
        frame_cnt = metadata['frame count']
        fps = metadata['frame rate']
        nb_short_videos = frame_cnt // (short_video_length*fps)
        # feature_name = 'Object Velocity'
        feature_name = 'Object Area'
        shvid_fts, _ = sample_video_features(video_features, metadata,
                                             short_video_length, 1)
        sampled_vals = list()
        sampled_medians = dict()
        sampled_means = dict()
        sampled_90_percent = dict()
        sampled_75_percent = dict()
        sampled_25_percent = dict()
        sampled_10_percent = dict()
        sampled_medians = dict()
        sampled_cv = dict()
        print('num of short vid={}'.format(len(shvid_fts.keys())))
        for i in range(nb_short_videos):
            processed_fts = nonzero(shvid_fts[i][feature_name])

            if processed_fts:
                sampled_vals.extend(processed_fts)
                sampled_medians[i] = np.median(processed_fts)
                sampled_means[i] = np.mean(processed_fts)
                sampled_90_percent[i] = np.quantile(processed_fts, 0.90)
                sampled_75_percent[i] = np.quantile(processed_fts, 0.75)
                sampled_25_percent[i] = np.quantile(processed_fts, 0.25)
                sampled_10_percent[i] = np.quantile(processed_fts, 0.1)
                sampled_cv[i] = np.std(processed_fts)/np.mean(processed_fts)
            else:
                sampled_means[i] = np.nan
                sampled_medians[i] = np.nan
                sampled_90_percent[i] = np.nan
                sampled_75_percent[i] = np.nan
                sampled_25_percent[i] = np.nan
                sampled_10_percent[i] = np.nan
                sampled_cv[i] = np.nan

        # load all awstream profile results
        _, resol_dict, acc_dict, size_dict, cnt_dict = \
            load_awstream_profile('/data/zxxia/benchmarking/awstream/spatial_overfitting_profile_11_06/awstream_spatial_overfitting_profile_{}.csv'.format(video),
                                  '/data/zxxia/benchmarking/awstream/short_video_size.csv')
        assert len(resol_dict) == len(acc_dict)
        assert len(resol_dict) == len(sampled_means), \
            "{} != {}".format(len(resol_dict), len(sampled_means))

        # plot 3D figure to indicate f1 vs feature at different fps
        plt.figure(0)
        for i in range(nb_short_videos):
            clip = video + '_' + str(i)
            axs.scatter(np.ones(len(resol_dict[clip]))*sampled_means[i],
                        resol_dict[clip], acc_dict[clip], s=5,
                        c=resolution_list, cmap=cmap)
        plt.xlim([0, 0.15])

        target_resol = 360
        for i in range(nb_short_videos):
            clip = video + '_' + str(i)
            # print(clip, sampled_means[i],
            #       sampled_25_percent[i], sampled_75_percent[i])
            for pf, ac, box_cnt in zip(resol_dict[clip],
                                       acc_dict[clip],
                                       cnt_dict[clip]):
                if box_cnt <= 200:
                    continue
                # if sampled_cv[i] > 0.71:
                #     continue
                # if clip not in filterd_videos:
                #     continue
                if pf == target_resol:
                    # print(pf, ac, sampled_means[i])
                    # plt.plot(sampled_means[i], ac)
                    acc_at_target_resol.append(ac)
                    # feat_to_plot.append(sampled_means[i])
                    # feat_to_plot.append(sampled_medians[i])
                    feat_to_plot.append(sampled_90_percent[i])
                    # feat_to_plot.append(sampled_75_percent[i])
                    # feat_to_plot.append(sampled_25_percent[i])
                    # feat_to_plot.append(sampled_10_percent[i])
                    # mobilenet_feat_to_plot.append(mobilenet_sampled_means[i])
                    annot_to_plot.append(clip)
                    # print(feat, pf, ac)
    plt.figure(1)
    plt.scatter(feat_to_plot, acc_at_target_resol, s=7, label='frcnn')
    # plt.scatter(mobilenet_feat_to_plot, acc_at_target_perf, s=7,
    #             label='mobilenet')
    plt.legend()
    plt.ylim([0, 1.1])
    # plt.xlim([0, 0.1])
    # for annot, feat, acc in zip(annot_to_plot, feat_to_plot,
    #                             acc_at_target_resol):
    #     plt.annotate(annot, (feat, acc), fontsize=5)
    plt.title("resolution={}".format(target_resol))
    plt.xlabel('Object Size Mean')
    plt.ylabel('accuracy')
    # plt.axhline(y=0.85, c='k', linestyle='--')
    # plt.axhline(y=0.50, c='k', linestyle='--')
    # plt.axvline(x=1.40, c='k', linestyle='--')
    # plt.axvline(x=2.5, c='k', linestyle='--')
    # # print(len(features), len(acc_at_target_perf))
    plt.show()


if __name__ == '__main__':
    main()


# use mobilenet detections to compute feature
# shvid_fts, _ = sample_video_features(mobilenet_video_features,
#                                      metadata, short_video_length,
#                                      n_frame)
# mobilenet_sampled_vals = list()
# mobilenet_sampled_medians = dict()
# mobilenet_sampled_means = dict()
# mobilenet_sampled_75_percent = dict()
# mobilenet_sampled_25_percent = dict()
# mobilenet_sampled_medians = dict()
# print('num of short vid={}'.format(len(shvid_fts.keys())))
# for i in range(nb_short_videos):
#     processed_fts = nonzero(shvid_fts[i][feature_name])
#
#     if processed_fts:
#         mobilenet_sampled_vals.extend(processed_fts)
#         mobilenet_sampled_medians[i] = np.median(processed_fts)
#         mobilenet_sampled_means[i] = np.mean(processed_fts)
#         mobilenet_sampled_75_percent[i] = np.quantile(processed_fts, 0.75)
#         mobilenet_sampled_25_percent[i] = np.quantile(processed_fts, 0.25)
#     else:
#         mobilenet_sampled_means[i] = np.nan
#         mobilenet_sampled_medians[i] = np.nan
#         mobilenet_sampled_75_percent[i] = np.nan
#         mobilenet_sampled_25_percent[i] = np.nan
