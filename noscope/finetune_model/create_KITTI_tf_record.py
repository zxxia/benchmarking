
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import json
import os
import contextlib2
import numpy as np
import PIL.Image
import glob
from create_youtube_tf_record import create_tf_example

import tensorflow as tf
import sys
from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
# from object_detection.utils import label_map_util
from benchmarking.video import YoutubeVideo, KittiVideo
from benchmarking.utils.model_utils import load_COCOlabelmap

ROOT = '/mnt/data/zhujun/dataset/KITTI/'
# KITTI_datapath = '/mnt/data/zhujun/dataset/KITTI/KITTI/City/2011_09_26_drive_0001_sync/image_02/data/720p/profile/
ORIGINAL_RESOL = '720p'
LOCATIONS = ['City', 'Residential', 'Road']
dataset_name = 'KITTI'
output_path = os.path.join('/mnt/data/zhujun/dataset/NoScope_finetuned_models', 'KITTI', 'data')
num_shards = 1
include_masks = False
if not os.path.exists(output_path):
    os.makedirs(output_path)

train_output_path = os.path.join(output_path, dataset_name + '_train.record')
val_output_path = os.path.join(output_path, dataset_name + '_val.record')
missing_annotation_count = 0
total_num_annotations_skipped = 0

with contextlib2.ExitStack() as tf_record_close_stack:
    output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
        tf_record_close_stack, train_output_path, num_shards)
    coco_map_path = '../mscoco_label_map.pbtxt'
    category_index = load_COCOlabelmap(coco_map_path)


    for loc in ['City']:
        for seg_path in sorted(glob.glob(os.path.join(ROOT, loc, '*'))):
            print(seg_path)
            if not os.path.isdir(seg_path):
                continue
            video_name = loc + '_' + os.path.basename(seg_path)
            print(seg_path)
            # loading videos
            img_path = os.path.join(
                seg_path, 'image_02', 'data', ORIGINAL_RESOL)
            dt_file = os.path.join(
                seg_path, 'image_02', 'data', ORIGINAL_RESOL, 'profile',
                'updated_gt_FasterRCNN_COCO_no_filter.csv')
            video = KittiVideo(
                video_name, ORIGINAL_RESOL, dt_file, img_path,
                filter_flag=True, merge_label_flag=True)

            annotations_index = video.get_video_detection()
            resol = video.resolution
            start_frame = video.start_frame_index
            end_frame = video.end_frame_index            
            for index in range(start_frame, end_frame):
                image = {}
                image['height'] = resol[1]
                image['width'] = resol[0]
                image['filename'] = '{:010d}.png'.format(index)      
                image['id'] = 'video_name_{:010d}.png'.format(index) 
                # if index not in annotations_index:
                #   missing_annotation_count += 1
                #   annotations_index[index] = []
                assert index in annotations_index, print('frame {} not in annotation'.format(index))
                if index % 100 == 0:
                    tf.logging.info('On image %d of %d', index, end_frame - start_frame)
                    annotations_list = annotations_index[index]
                _, tf_example, num_annotations_skipped = create_tf_example(
                    image, annotations_list, img_path, category_index, include_masks)
                total_num_annotations_skipped += num_annotations_skipped
                shard_idx = index % num_shards
                output_tfrecords[shard_idx].write(tf_example.SerializeToString())
tf.logging.info('Finished writing, skipped %d annotations.',
                    total_num_annotations_skipped)



missing_annotation_count = 0
total_num_annotations_skipped = 0
with contextlib2.ExitStack() as tf_record_close_stack:
    output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
        tf_record_close_stack, val_output_path, num_shards)
    coco_map_path = '../mscoco_label_map.pbtxt'
    category_index = load_COCOlabelmap(coco_map_path)


    for loc in ['Road']:
        for seg_path in sorted(glob.glob(os.path.join(ROOT, loc, '*'))):
            print(seg_path)
            if not os.path.isdir(seg_path):
                continue
            video_name = loc + '_' + os.path.basename(seg_path)
            print(seg_path)
            # loading videos
            img_path = os.path.join(
                seg_path, 'image_02', 'data', ORIGINAL_RESOL)
            dt_file = os.path.join(
                seg_path, 'image_02', 'data', ORIGINAL_RESOL, 'profile',
                'updated_gt_FasterRCNN_COCO_no_filter.csv')
            video = KittiVideo(
                video_name, ORIGINAL_RESOL, dt_file, img_path,
                filter_flag=True, merge_label_flag=True)

            annotations_index = video.get_video_detection()
            resol = video.resolution
            start_frame = video.start_frame_index
            end_frame = video.end_frame_index            
            for index in range(start_frame, end_frame):
                image = {}
                image['height'] = resol[1]
                image['width'] = resol[0]
                image['filename'] = '{:010d}.png'.format(index)      
                image['id'] = 'video_name_{:010d}.png'.format(index) 
                # if index not in annotations_index:
                #   missing_annotation_count += 1
                #   annotations_index[index] = []
                assert index in annotations_index, print('frame {} not in annotation'.format(index))
                if index % 100 == 0:
                    tf.logging.info('On image %d of %d', index, end_frame - start_frame)
                annotations_list = annotations_index[index]
                _, tf_example, num_annotations_skipped = create_tf_example(
                    image, annotations_list, img_path, category_index, include_masks)
                total_num_annotations_skipped += num_annotations_skipped
                shard_idx = index % num_shards
                output_tfrecords[shard_idx].write(tf_example.SerializeToString())
tf.logging.info('Finished writing, skipped %d annotations.',
                    total_num_annotations_skipped)


