import csv
import glob
import os
import numpy as np
from benchmarking.feature_analysis.features import compute_velocity, \
    compute_video_object_size, compute_arrival_rate, \
    compute_percentage_frame_with_object, count_unique_class
from benchmarking.video import YoutubeVideo, WaymoVideo
from benchmarking.utils.model_utils import load_full_model_detection

VIDEOS = ['crossroad', 'crossroad2', 'crossroad3', 'crossroad4', 'drift',
          'driving1', 'driving_downtown', 'highway',
          'nyc', 'jp',  'lane_split',  'driving2',
          'motorway', 'park', 'russia', 'russia1', 'traffic', 'tw', 'tw1',
          'tw_under_bridge']

DT_ROOT = '/data/zxxia/benchmarking/results/videos'
SHORT_VIDEO_LENGTH = 30


for name in VIDEOS:
    with open('video_features_30s/{}_features.csv'.format(name), 'w', 1) as f:
        writer = csv.writer(f)
        writer.writerow(['name', 'average velocity',
                         'average object size', 'average total object size',
                         'percentage of frame with object',
                         'number of distinct classes'])
        print(name)
        dt_file = os.path.join(
            DT_ROOT, name, '720p',
            'profile/updated_gt_FasterRCNN_COCO_no_filter.csv')
        metadata_file = '/data/zxxia/videos/{}/metadata.json'.format(name)
        video = YoutubeVideo(name, '720p', metadata_file, dt_file, None)
        velocity = compute_velocity(
            video.get_video_detection(), video.start_frame_index,
            video.end_frame_index, video.frame_rate)
        arrival_rate = compute_arrival_rate(
            video.get_video_detection(), video.start_frame_index,
            video.end_frame_index, video.frame_rate)
        obj_size, tot_obj_size = compute_video_object_size(
            video.get_video_detection(), video.start_frame_index,
            video.end_frame_index, video.resolution)
        chunk_frame_cnt = SHORT_VIDEO_LENGTH * video.frame_rate
        nb_chunks = video.frame_count // chunk_frame_cnt

        for i in range(nb_chunks):
            clip = name + '_' + str(i)
            start_frame = i * chunk_frame_cnt + video.start_frame_index
            end_frame = (i + 1) * chunk_frame_cnt
            velo = []
            arr_rate = []
            sizes = []
            tot_sizes = []
            percent_with_obj = compute_percentage_frame_with_object(
                video.get_video_detection(), start_frame, end_frame)
            nb_distinct_classes = count_unique_class(
                video.get_video_detection(), start_frame, end_frame)
            for j in range(start_frame, end_frame+1):
                velo.extend(velocity[j])
                arr_rate.append(arrival_rate[j])
                sizes.extend(obj_size[j])
                tot_sizes.append(tot_obj_size[j])
            writer.writerow(
                [clip, np.mean(velo), np.mean(sizes),
                 np.mean(tot_sizes), percent_with_obj, nb_distinct_classes])


for name in ['cropped_crossroad4', 'cropped_crossroad4_2',
             'cropped_crossroad5', 'cropped_driving2']:
    with open('video_features_30s/{}_features.csv'.format(name), 'w', 1) as f:
        writer = csv.writer(f)
        writer.writerow(['name', 'average velocity',
                         'average object size', 'average total object size',
                         'percentage of frame with object',
                         'number of distinct classes'])
        dt_file = os.path.join(
            DT_ROOT, name, 'updated_gt_FasterRCNN_COCO_no_filter.csv')
        dets, num_of_frames = load_full_model_detection(dt_file)
        print(dets)
        velocity = compute_velocity(dets, min(dets), max(dets), 30)
        arrival_rate = compute_arrival_rate(dets, min(dets), max(dets), 30)
        obj_size, tot_obj_size = compute_video_object_size(
            dets, min(dets), max(dets), (600, 400))
        chunk_frame_cnt = SHORT_VIDEO_LENGTH * 30
        nb_chunks = num_of_frames // chunk_frame_cnt

        for i in range(nb_chunks):
            clip = name + '_' + str(i)
            start_frame = i * chunk_frame_cnt + min(dets)
            end_frame = (i + 1) * chunk_frame_cnt
            velo = []
            arr_rate = []
            sizes = []
            tot_sizes = []
            percent_with_obj = compute_percentage_frame_with_object(
                dets, start_frame, end_frame)
            nb_distinct_classes = count_unique_class(
                dets, start_frame, end_frame)
            for j in range(start_frame, end_frame+1):
                velo.extend(velocity[j])
                arr_rate.append(arrival_rate[j])
                sizes.extend(obj_size[j])
                tot_sizes.append(tot_obj_size[j])
            writer.writerow(
                [clip, np.mean(velo), np.mean(sizes),
                 np.mean(tot_sizes), percent_with_obj, nb_distinct_classes])

WAYMO_ROOT = '/data/zxxia/ekya/datasets/waymo_images'
with open('video_features_30s/waymo_features.csv', 'w', 1) as f:
    writer = csv.writer(f)
    writer.writerow(['name', 'average velocity',
                     'average object size', 'average total object size',
                     'percentage of frame with object',
                     'number of distinct classes'])
    for seg_path in glob.glob(os.path.join(WAYMO_ROOT, '*')):
        name = os.path.basename(seg_path)
        print(name)
        dt_file = os.path.join(seg_path, 'FRONT/profile',
                               'updated_gt_FasterRCNN_COCO_no_filter.csv')
        video = WaymoVideo(name, '720p', dt_file, None)
        velocity = compute_velocity(
            video.get_video_detection(), video.start_frame_index,
            video.end_frame_index, video.frame_rate)
        arrival_rate = compute_arrival_rate(
            video.get_video_detection(), video.start_frame_index,
            video.end_frame_index, video.frame_rate)
        obj_size, tot_obj_size = compute_video_object_size(
            video.get_video_detection(), video.start_frame_index,
            video.end_frame_index, video.resolution)
        chunk_frame_cnt = SHORT_VIDEO_LENGTH * video.frame_rate
        nb_chunks = video.frame_count // chunk_frame_cnt

        velo = []
        arr_rate = []
        sizes = []
        tot_sizes = []
        percent_with_obj = compute_percentage_frame_with_object(
            video.get_video_detection(), video.start_frame_index,
            video.end_frame_index)
        nb_distinct_classes = count_unique_class(
            video.get_video_detection(), video.start_frame_index,
            video.end_frame_index)
        for j in range(video.start_frame_index, video.end_frame_index+1):
            velo.extend(velocity[j])
            arr_rate.append(arrival_rate[j])
            sizes.extend(obj_size[j])
            tot_sizes.append(tot_obj_size[j])
        writer.writerow(
            [name, np.mean(velo), np.mean(sizes),
             np.mean(tot_sizes), percent_with_obj, nb_distinct_classes])
