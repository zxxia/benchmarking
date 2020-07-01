import csv
import os
import glob
import pdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from benchmarking.video import YoutubeVideo, WaymoVideo
from benchmarking.feature_analysis.features import compute_velocity, \
    compute_box_size, compute_video_object_size
import time


class FeatureScanner():
    def scan_object_velocity(self):
        pass

    def scan_object_size(self):
        pass
# np.random.seed(1)


# from benchmarking.utils.model_utils import
#     filter_video_detections, remove_overlappings

DT_ROOT = '/data/zxxia/benchmarking/results/videos'
# VIDEOS = ['crossroad', 'crossroad2', 'crossroad3', 'crossroad4', 'drift',
#           'driving1', 'driving_downtown', 'highway',
#           'nyc', 'jp',  'lane_split',  'driving2',
#           'motorway', 'park', 'russia', 'russia1', 'traffic', 'tw', 'tw1',
#           'tw_under_bridge']
VIDEOS = sorted(['crossroad', 'crossroad2', 'crossroad3', 'crossroad4',
                 'driving1', 'driving_downtown',  # 'lane_split',
                 'highway', 'driving2',  # 'jp', drift
                 # 'jp_hw','highway_normal_traffic'
                 'motorway', 'nyc', 'park',
                 'russia1',
                 'traffic',
                 # 'tw', 'tw1',  # 'tw_road', 'tw_under_bridge',
                 # 'road_trip'
                 ])
# SHORT_VIDEO_LENGTH = 10
SHORT_VIDEO_LENGTH = 30
PROFILE_LENGTH = 0

SAMPLE_STEP_LIST = [1, 2, 5, 8, 10, 12, 15, 18,
                    20, 25, 30, 35, 40, 45, 50, 100]
# SAMPLE_STEP_LIST = [1]
# RESOL_LIST = ['720p', '540p', '480p', '360p']  # , '180p']
RESOL_LIST = ['480p']  # , '180p']
# RESOL_LIST = ['360p']  # , '180p']
ORIGINAL_REOSL = '720p'
# VIDEOS = ['park', 'nyc']
VIDEOS = ['road_trip']

GOOD_CASE_TH = 0.85
BAD_CASE_TH = 0.78
# BAD_CASE_TH = 0.70

step = 0.004
area_bins = np.arange(0, 1, step)


def scan_object_velocity(video, sample_step=10):
    scanned_velocity = compute_velocity(video.get_video_detection(),
                                        video.start_frame_index,
                                        video.end_frame_index,
                                        video.frame_rate,
                                        sample_step=sample_step)
    nb_short_videos = video.frame_count//(SHORT_VIDEO_LENGTH*video.frame_rate)
    print(nb_short_videos, video.video_name)
    short_video_velocities = []
    for i in range(nb_short_videos):
        clip = video.video_name + '_' + str(i)
        start = i*SHORT_VIDEO_LENGTH*video.frame_rate + \
            video.start_frame_index + PROFILE_LENGTH * video.frame_rate
        end = (i+1)*SHORT_VIDEO_LENGTH*video.frame_rate + \
            video.start_frame_index
        velos = []
        for j in range(start, end+1):
            if j in scanned_velocity:
                velos.extend(scanned_velocity[j])
        short_video_velocities.append(velos)
    return short_video_velocities


def scan_object_size(original_video, video, benchmark_f1, area_bins,
                     sample_step=10):
    """Do feature scanning on object size."""
    nb_short_videos = original_video.frame_count//(
        SHORT_VIDEO_LENGTH*original_video.frame_rate)
    print(nb_short_videos, video.video_name)
    # return None, None, None
    # print(nb_short_videos)
    # gt_f1 = []
    scan_f1 = []
    scan_f1_mobilenet = []
    scan_f1_mobilenet_sampled = []

    start_t = time.time()
    for i in range(nb_short_videos):
        clip = video.video_name + '_' + str(i)
        start = i*SHORT_VIDEO_LENGTH*original_video.frame_rate + \
            original_video.start_frame_index + PROFILE_LENGTH * original_video.frame_rate
        end = (i+1)*SHORT_VIDEO_LENGTH*original_video.frame_rate + \
            original_video.start_frame_index
        cnt = np.zeros(len(area_bins)-1)
        cnt_mobilenet = np.zeros(len(area_bins)-1)
        cnt_mobilenet_sampled = np.zeros(len(area_bins)-1)

        filename = './scanned_features/{}_{}_obj_size_{}.pkl'.format(
            original_video.video_name, i, video.resolution[1])
        if os.path.exists(filename):
            areas = pickle.load(open(filename, 'rb'))
        else:
            areas, _ = compute_video_object_size(
                original_video.get_video_detection(), start, end,
                original_video.resolution, sample_step=1)
            pickle.dump(areas, open(filename, 'wb'))
        filename = './scanned_features/{}_{}_obj_size_mobilenet.pkl'.format(
            original_video.video_name, i, video.resolution[1])
        if os.path.exists(filename):
            areas_mobilenet = pickle.load(open(filename, 'rb'))
        else:
            areas_mobilenet, _ = compute_video_object_size(
                video.get_video_detection(), start, end, video.resolution,
                sample_step=sample_step)
            pickle.dump(areas_mobilenet, open(filename, 'wb'))
        filename = './scanned_features/{}_{}_obj_size_mobilenet_{}_sample_step_{}.pkl'.format(
            original_video.video_name, i, video.resolution[1], sample_step)
        if os.path.exists(filename):
            areas_mobilenet_sampled = pickle.load(open(filename, 'rb'))
        else:
            areas_mobilenet_sampled, _ = compute_video_object_size(
                video.get_video_detection(), start, end, video.resolution,
                sample_step=sample_step)
            pickle.dump(areas_mobilenet_sampled, open(filename, 'wb'))
        # print(clip, 'nb of box:', len(areas))
        # if len(areas) < 500:
        #     continue
        # pdb.set_trace()
        areas = [val for frame_idx in areas for val in areas[frame_idx]]
        areas_mobilenet = [
            val for frame_idx in areas_mobilenet for val in areas_mobilenet[frame_idx]]
        areas_mobilenet_sampled = [
            val for frame_idx in areas_mobilenet_sampled for val in areas_mobilenet_sampled[frame_idx]]
        # print(sample_step, len(areas_mobilenet_sampled))
        for area in areas:
            # print(area)
            for j in range(1, len(area_bins)):
                if area_bins[j-1] <= area < area_bins[j]:
                    cnt[j-1] += 1
                    break

        for area in areas_mobilenet:
            for j in range(1, len(area_bins)):
                if area_bins[j-1] <= area < area_bins[j]:
                    cnt_mobilenet[j-1] += 1
                    break

        for area in areas_mobilenet_sampled:
            for j in range(1, len(area_bins)):
                if area_bins[j-1] <= area < area_bins[j]:
                    cnt_mobilenet_sampled[j-1] += 1
                    break

        cnt_percent = cnt/np.sum(cnt)
        pred_f1 = np.dot(np.nan_to_num(benchmark_f1),
                         np.nan_to_num(cnt_percent))
        scan_f1.append(pred_f1)

        cnt_percent_mobilenet = cnt_mobilenet/np.sum(cnt_mobilenet)
        pred_f1_mobilenet = np.dot(np.nan_to_num(benchmark_f1),
                                   np.nan_to_num(cnt_percent_mobilenet))
        scan_f1_mobilenet.append(pred_f1_mobilenet)

        # print(np.sum(cnt_mobilenet_sampled))
        cnt_percent_mobilenet_sampled = cnt_mobilenet_sampled / \
            np.sum(cnt_mobilenet_sampled)
        pred_f1_mobilenet_sampled = np.dot(
            np.nan_to_num(benchmark_f1),
            np.nan_to_num(cnt_percent_mobilenet_sampled))
        scan_f1_mobilenet_sampled.append(pred_f1_mobilenet_sampled)
    print('finish scanning and use {}s'.format(time.time()-start_t))
    return scan_f1, scan_f1_mobilenet, scan_f1_mobilenet_sampled


# Load all youtube video features computed based on FasterRCNN
# Resnet101 detections

# name = 'road_trip'
# profile_data = pd.read_csv(
#     '/data/zxxia/benchmarking/results/awstream/overfitting_results_30s_10s/awstream_spatial_overfitting_profile_{}.csv'.format(name))
# dt_file = os.path.join(DT_ROOT, name, '720p', 'profile',
#                        'updated_gt_FasterRCNN_COCO_no_filter.csv')
# # dt_file = None
# if name == 'road_trip':
#     metadata_file = '/data2/zxxia/videos/{}/metadata.json'.format(name)
# else:
#     metadata_file = '/data/zxxia/videos/{}/metadata.json'.format(name)
# start_t = time.time()
# filename = './scanned_features/{}_obj.pkl'.format(name)
# if os.path.exists(filename):
#     original_video = pickle.load(open(filename, 'rb'))
# else:
#     original_video = YoutubeVideo(name, '720p', metadata_file, dt_file, None)
#     pickle.dump(original_video, open(filename, 'wb'))
#
# dt_file = os.path.join(DT_ROOT, name, '720p', 'profile',
#                        'updated_gt_mobilenet_COCO_no_filter.csv')
# metadata_file = '/data/zxxia/videos/{}/metadata.json'.format(name)
# resol = RESOL_LIST[0]
# filename = './scanned_features/{}_obj_mobilenet_{}.pkl'.format(
#     name, resol)
# if os.path.exists(filename):
#     video = pickle.load(open(filename, 'rb'))
# else:
#     video = YoutubeVideo(name, resol, metadata_file, dt_file, None)
#     pickle.dump(video, open(filename, 'wb'))
# print('video finish loading and use {}s..'.format(time.time()-start_t))
# gt_f1 = profile_data.loc[profile_data['resolution'] == resol]['f1'].to_numpy()
# print(gt_f1.shape)
# benchmark_f1 = np.load('f1_vs_obj_sizes/{}_binwise_f1.npy'.format(resol))
# estimation_error_mean_dict = {}
# estimation_error_std_dict = {}
# baseline_estimation_error_mean_dict = {}
# baseline_estimation_error_std_dict = {}
# bad_estimation_error_mean_dict = {}
# bad_estimation_error_std_dict = {}
# bad_baseline_estimation_error_mean_dict = {}
# bad_baseline_estimation_error_std_dict = {}
# for round_idx in range(300):
#     # profile_data = pd.read_csv(
#     #     '~/Projects/benchmarking/awstream/overfitting_results_30s_10s/awstream_spatial_overfitting_profile_{}.csv'.format(name))
#     # profile_data = pd.read_csv(
#     #     '/data/zxxia/benchmarking/awstream/spatial_overfitting_profile_10s/awstream_spatial_overfitting_profile_{}.csv'.format(name))
#     # sample_step_estimation_errors = {}
#     # sample_step_bad_estimation_errors = {}
#     # sample_step_baseline_estimation_errors = {}
#     # sample_step_bad_baseline_estimation_errors = {}
#     estimation_errors = []
#     baseline_estimation_errors = []
#     bad_estimation_errors = []
#     bad_baseline_estimation_errors = []
#     for sample_step in SAMPLE_STEP_LIST:
#         filename = './scanned_features/scan_f1_mobilenet_sampled_{}.pkl'.format(
#             sample_step)
#         if os.path.exists(filename):
#             scan_f1_mobilenet_sampled = pickle.load(open(filename, 'rb'))
#         else:
#             scan_f1, scan_f1_mobilenet, scan_f1_mobilenet_sampled = scan_object_size(
#                 original_video, video, benchmark_f1, area_bins, sample_step)
#             pickle.dump(scan_f1_mobilenet_sampled, open(filename, 'wb'))
#
#         nb_baseline = int(gt_f1.shape[0]/sample_step)
#         print(sample_step, nb_baseline)
#         gt_good_percent = np.sum(gt_f1 >= GOOD_CASE_TH)/gt_f1.shape[0]
#         gt_bad_percent = np.sum(gt_f1 <= BAD_CASE_TH)/gt_f1.shape[0]
#         scan_good_percent = np.sum(
#             np.array(scan_f1_mobilenet_sampled) >= GOOD_CASE_TH)/gt_f1.shape[0]
#         scan_bad_percent = np.sum(
#             np.array(scan_f1_mobilenet_sampled) <= BAD_CASE_TH)/gt_f1.shape[0]
#         if nb_baseline == gt_f1.shape[0]:
#             baseline_start_index = 0
#         else:
#             baseline_start_index = np.random.choice(
#                 gt_f1.shape[0] - nb_baseline)
#         baseline_f1 = gt_f1[baseline_start_index: baseline_start_index + nb_baseline]
#         baseline_good_percent = np.sum(
#             baseline_f1 >= GOOD_CASE_TH)/baseline_f1.shape[0]
#         baseline_bad_percent = np.sum(
#             baseline_f1 <= BAD_CASE_TH)/baseline_f1.shape[0]
#         print('gt', sample_step, resol, name, gt_good_percent, gt_bad_percent)
#         print('scan', sample_step, resol, name,
#               scan_good_percent, scan_bad_percent)
#         print('baseline', sample_step, resol, name, baseline_good_percent,
#               baseline_bad_percent)
#         # print(np.array(scan_f1_mobilenet_sampled)-gt_f1)
#         estimation_errors.append(np.abs(gt_good_percent - scan_good_percent))
#         baseline_estimation_errors.append(
#             np.abs(gt_good_percent - baseline_good_percent))
#         bad_estimation_errors.append(np.abs(gt_bad_percent - scan_bad_percent))
#         bad_baseline_estimation_errors.append(
#             np.abs(gt_bad_percent - baseline_bad_percent))
#     # estimation_error_mean_dict[sample_step] = np.mean(estimation_errors)
#     # estimation_error_std_dict[sample_step] = np.std(estimation_errors)
#     # bad_estimation_error_mean_dict[sample_step] = np.mean(
#     #     bad_estimation_errors)
#     # bad_estimation_error_std_dict[sample_step] = np.std(bad_estimation_errors)
#     # baseline_estimation_error_mean_dict[sample_step] = np.mean(
#     #     baseline_estimation_errors)
#     # baseline_estimation_error_std_dict[sample_step] = np.std(
#     #     baseline_estimation_errors)
#     # bad_baseline_estimation_error_mean_dict[sample_step] = np.mean(
#     #     bad_baseline_estimation_errors)
#     # bad_baseline_estimation_error_std_dict[sample_step] = np.std(
#     #     bad_baseline_estimation_errors)
#     with open(f'./round_ft_scan_results/{name}_aws_scan_error_round{round_idx}.csv', 'w', 1) as f:
#         writer = csv.writer(f)
#         writer.writerow(
#             ['sample_step', 'estimation error', 'bad estimation error',
#              'baseline estimation error', 'bad baseline estimation error'])
#         for idx, sample_step in enumerate(SAMPLE_STEP_LIST):
#             writer.writerow(
#                 [sample_step, estimation_errors[idx],
#                  bad_estimation_errors[idx],
#                  baseline_estimation_errors[idx],
#                  bad_baseline_estimation_errors[idx]])

# with open('spatial_scan_error.csv', 'w', 1) as f:
#     writer = csv.writer(f)
#     writer.writerow(
#         ['resolution', 'estimation error mean', 'estimation error std',
#          'bad estimation error mean', 'bad estimation error std',
#          'baseline estimation error mean', 'baseline estimation error std',
#          'bad baseline estimation error mean',
#          'bad baseline estimation error std'])
#     for sample_step in SAMPLE_STEP_LIST:
#         writer.writerow(
#             [sample_step,
#              estimation_error_mean_dict[sample_step],
#              estimation_error_std_dict[sample_step],
#              bad_estimation_error_mean_dict[sample_step],
#              bad_estimation_error_std_dict[sample_step],
#              baseline_estimation_error_mean_dict[sample_step],
#              baseline_estimation_error_std_dict[sample_step],
#              bad_baseline_estimation_error_mean_dict[sample_step],
#              bad_baseline_estimation_error_std_dict[sample_step]])
# print(estimation_error_mean_dict)
# print(estimation_error_std_dict)

SAMPLE_STEP_LIST = [1, 2, 10, 15, 20,
                    25, 30, 35, 40, 50, 80, 100, 125, 140, 200, 300, 400]
SHORT_VIDEO_LENGTH = 30
BAD_CASE_TH = 0.700
estimation_error_mean_dict = {}
estimation_error_std_dict = {}
baseline_estimation_error_mean_dict = {}
baseline_estimation_error_std_dict = {}
bad_estimation_error_mean_dict = {}
bad_estimation_error_std_dict = {}
bad_baseline_estimation_error_mean_dict = {}
bad_baseline_estimation_error_std_dict = {}
for round_idx in range(300):
    sample_step_estimation_errors = dict()
    sample_step_baseline_estimation_errors = dict()
    sample_step_bad_estimation_errors = dict()
    sample_step_bad_baseline_estimation_errors = dict()
    for sample_step in SAMPLE_STEP_LIST:
        estimation_errors = []
        baseline_estimation_errors = []
        bad_estimation_errors = []
        bad_baseline_estimation_errors = []
        # for name in ['road_trip', 'driving_downtown',  'nyc', 'lane_split']:  # 'park',
        for name in ['road_trip']:  # 'park',
            if os.path.exists(f'scanned_features/{name}_{sample_step}.pkl'):
                scanned_velocity = pickle.load(
                    open(f'scanned_features/{name}_{sample_step}.pkl', 'rb'))
                # pdb.set_trace()
            else:
                dt_file = os.path.join(DT_ROOT, name, '720p', 'profile',
                                       'updated_gt_mobilenet_COCO_no_filter.csv')
                metadata_file = '/data/zxxia/videos/{}/metadata.json'.format(
                    name)
                video = YoutubeVideo(
                    name, '720p', metadata_file, dt_file, None)
                scanned_velocity = scan_object_velocity(
                    video, sample_step=sample_step)
                pickle.dump(scanned_velocity, open(
                    f'scanned_features/{name}_{sample_step}.pkl', 'wb'))
            # print(scanned_velocity)
            # profile_data = pd.read_csv(
            #     '~/Projects/benchmarking/awstream/overfitting_results_30s_10s/awstream_spatial_overfitting_profile_{}.csv'.format(name))
            profile_data = pd.read_csv(
                '/data/zxxia/benchmarking/videostorm/overfitting_profile_10_14/videostorm_overfitting_profile_{}.csv'.format(name))
            # profile_data = pd.read_csv(
            #     f'/data/zxxia/benchmarking/results/videostorm/test_coverage_profile/videostorm_coverage_profile_{name}.csv')
            gt = profile_data[profile_data['frame_rate']
                              == 0.2]['f1'].to_numpy()
            # print(gt)
            scanned_avg_velo = np.array([np.mean(velos)
                                         for velos in scanned_velocity])
            gt_good_percent = np.sum(gt >= GOOD_CASE_TH) / gt.shape[0]
            gt_bad_percent = np.sum(gt <= BAD_CASE_TH) / gt.shape[0]
            scan_good_percent = np.sum(
                scanned_avg_velo <= 1.45) / scanned_avg_velo.shape[0]  # 1.4
            scan_bad_percent = np.sum(
                scanned_avg_velo >= 1.6)/scanned_avg_velo.shape[0]
            if sample_step == 1:
                # nb_baseline = gt.shape[0]
                nb_baseline = int(round(gt.shape[0]/sample_step))
            else:
                nb_baseline = int(round(gt.shape[0]/sample_step))
            print(nb_baseline)
            print(gt.shape)
            if nb_baseline == gt.shape[0]:
                baseline_start_index = 0
            else:
                baseline_start_index = np.random.choice(
                    gt.shape[0] - nb_baseline)
            baseline_good_percent = np.sum(
                gt[baseline_start_index:baseline_start_index+nb_baseline] >= GOOD_CASE_TH) / nb_baseline
            baseline_bad_percent = np.sum(
                gt[baseline_start_index:baseline_start_index+nb_baseline] <= BAD_CASE_TH) / nb_baseline
            print(f'video:{name}, sample every {sample_step}')
            print(
                f'gt good percent:{gt_good_percent}, bad percent: {gt_bad_percent}')
            print(
                f'scan good percent: {scan_good_percent}, bad percent: {scan_bad_percent}')
            print(
                f'baseline good percent: {baseline_good_percent}, bad percent: {baseline_bad_percent}, baseline count:{nb_baseline}')
            sample_step_estimation_errors[sample_step] = np.abs(
                gt_good_percent - scan_good_percent)
            sample_step_baseline_estimation_errors[sample_step] = np.abs(
                gt_good_percent - baseline_good_percent)
            sample_step_bad_estimation_errors[sample_step] = np.abs(
                gt_bad_percent - scan_bad_percent)
            sample_step_bad_baseline_estimation_errors[sample_step] = np.abs(
                gt_bad_percent - baseline_bad_percent)

            estimation_errors.append(
                np.abs(gt_good_percent - scan_good_percent))
            baseline_estimation_errors.append(
                np.abs(gt_good_percent - baseline_good_percent))
            bad_estimation_errors.append(
                np.abs(gt_bad_percent - scan_bad_percent))
            bad_baseline_estimation_errors.append(
                np.abs(gt_bad_percent - baseline_bad_percent))
        estimation_error_mean_dict[sample_step] = np.mean(estimation_errors)
        estimation_error_std_dict[sample_step] = np.std(estimation_errors)
        bad_estimation_error_mean_dict[sample_step] = np.mean(
            bad_estimation_errors)
        bad_estimation_error_std_dict[sample_step] = np.std(
            bad_estimation_errors)
        baseline_estimation_error_mean_dict[sample_step] = np.mean(
            baseline_estimation_errors)
        baseline_estimation_error_std_dict[sample_step] = np.std(
            baseline_estimation_errors)
        bad_baseline_estimation_error_mean_dict[sample_step] = np.mean(
            bad_baseline_estimation_errors)
        bad_baseline_estimation_error_std_dict[sample_step] = np.std(
            bad_baseline_estimation_errors)
    with open(f'./round_ft_scan_results/{name}_scan_error_round{round_idx}.csv', 'w', 1) as f:
        writer = csv.writer(f)
        writer.writerow(
            ['sample_step', 'estimation error',
             'bad estimation error',
             'baseline estimation error',
             'bad baseline estimation error'])
        for sample_step in SAMPLE_STEP_LIST:
            writer.writerow(
                [sample_step, sample_step_estimation_errors[sample_step],
                 sample_step_bad_estimation_errors[sample_step],
                 sample_step_baseline_estimation_errors[sample_step],
                 sample_step_bad_baseline_estimation_errors[sample_step]])


# with open('temporal_scan_error_new.csv', 'w', 1) as f:
#     writer = csv.writer(f)
#     writer.writerow(
#         ['sample_step', 'estimation error mean', 'estimation error std',
#          'bad estimation error mean', 'bad estimation error std',
#          'baseline estimation error mean', 'baseline estimation error std',
#          'bad baseline estimation error mean',
#          'bad baseline estimation error std'])
#     for sample_step in SAMPLE_STEP_LIST:
#         writer.writerow(
#             [sample_step,
#              estimation_error_mean_dict[sample_step],
#              estimation_error_std_dict[sample_step],
#              bad_estimation_error_mean_dict[sample_step],
#              bad_estimation_error_std_dict[sample_step],
#              baseline_estimation_error_mean_dict[sample_step],
#              baseline_estimation_error_std_dict[sample_step],
#              bad_baseline_estimation_error_mean_dict[sample_step],
#              bad_baseline_estimation_error_std_dict[sample_step]])


# pdb.set_trace()

# velocity = compute_velocity(
#     video.get_video_detection(), video.start_frame_index,
#     video.end_frame_index, video.frame_rate)
# chunk_frame_cnt = SHORT_VIDEO_LENGTH * video.frame_rate
# nb_chunks = video.frame_count // chunk_frame_cnt
# # print(velocity)
#
# for i in range(nb_chunks):
#     clip = name + '_' + str(i)
#     start_frame = i * chunk_frame_cnt + video.start_frame_index
#     end_frame = (i + 1) * chunk_frame_cnt
#     velo = []
#     for j in range(start_frame, end_frame+1):
#         if j in velocity:
#             velo.extend(velocity[j])

# Load all Waymo video features computed based on FasterRCNN
# Resnet101 detections
# ROOT = '/data/zxxia/ekya/datasets/waymo_images'
# for seg_path in glob.glob(os.path.join(ROOT, '*')):
#     # seg_name = os.path.basename(seg_path)
#     video_name = os.path.basename(seg_path)
#     print(video_name)
#     dt_file = os.path.join(seg_path, 'FRONT/profile',
#                            'updated_gt_FasterRCNN_COCO_no_filter.csv')
#     original_video = WaymoVideo(video_name, '720p', dt_file, None)
#     frame_rate = original_video.frame_rate
#     video = WaymoVideo(video_name, '720p', dt_file, None)
#     velocity = compute_velocity(
#         video.get_video_detection(), video.start_frame_index,
#         video.end_frame_index, video.frame_rate)
#     print(velocity)
