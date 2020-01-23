# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert raw COCO dataset to TFRecord for object_detection.

Please note that this tool creates sharded output files.

Example usage:
    python create_coco_tf_record.py --logtostderr \
      --train_image_dir="${TRAIN_IMAGE_DIR}" \
      --val_image_dir="${VAL_IMAGE_DIR}" \
      --test_image_dir="${TEST_IMAGE_DIR}" \
      --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
      --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
      --testdev_annotations_file="${TESTDEV_ANNOTATIONS_FILE}" \
      --output_dir="${OUTPUT_DIR}"
"""
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

import tensorflow as tf
import sys
from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
# from object_detection.utils import label_map_util
from benchmarking.video import YoutubeVideo
from benchmarking.utils.model_utils import load_COCOlabelmap

flags = tf.app.flags

flags.DEFINE_string('resol', '720p',
                    'Image resolution after resizing.')
flags.DEFINE_string('data_path', '', 'Data path.')
flags.DEFINE_string('dataset_name', '', 'Name of the dataset.')
flags.DEFINE_string('output_path', '', 'Dataset name.')
flags.DEFINE_string('start_frame', None, 'Start frame')
flags.DEFINE_string('end_frame', None, 'End fame(included).')
flags.DEFINE_string('train_range','1, 18001', 
                                  'Frame index range for training data.')
flags.DEFINE_string('val_range', '18001, 22001', 
                                  'Frame index range for val data.')
flags.DEFINE_boolean('include_masks', False,
                        'Whether to include instance segmentations masks '
                        '(PNG encoded) in the result. default: False.')
FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


def create_tf_example(image,
                      annotations_list,
                      image_dir,
                      category_index,
                      include_masks=False):
  """Converts image and annotations to a tf.Example proto.

  Args:
    image: dict with keys:
      [u'license', u'file_name', u'coco_url', u'height', u'width',
      u'date_captured', u'flickr_url', u'id']
    annotations_list:
      list of dicts with keys:
      [u'segmentation', u'area', u'iscrowd', u'image_id',
      u'bbox', u'category_id', u'id']
      Notice that bounding box coordinates in the official COCO dataset are
      given as [x, y, width, height] tuples using absolute coordinates where
      x, y represent the top-left (0-indexed) corner.  This function converts
      to the format expected by the Tensorflow Object Detection API (which is
      which is [ymin, xmin, ymax, xmax] with coordinates normalized relative
      to image size).
    image_dir: directory containing the image files.
    category_index: a dict containing COCO category information keyed
      by the 'id' field of each category.  See the
      label_map_util.create_category_index function.
    include_masks: Whether to include instance segmentations masks
      (PNG encoded) in the result. default: False.
  Returns:
    example: The converted tf.Example
    num_annotations_skipped: Number of (invalid) annotations that were ignored.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  image_height = image['height']
  image_width = image['width']
  filename = image['filename']
  image_id = image['id']
  full_path = os.path.join(image_dir, filename)
  with tf.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  # encoded_jpg_io = io.BytesIO(encoded_jpg)
  # image = PIL.Image.open(encoded_jpg_io)
  key = hashlib.sha256(encoded_jpg).hexdigest()

  xmin = []
  xmax = []
  ymin = []
  ymax = []
  is_crowd = []
  category_names = []
  category_ids = []
  area = []
  encoded_mask_png = []
  num_annotations_skipped = 0
  for object_annotations in annotations_list:
    x = object_annotations[0]
    y = object_annotations[1]
    width = object_annotations[2] - object_annotations[0]
    height = object_annotations[3] - object_annotations[1]
    category_id = object_annotations[4]
    if width <= 0 or height <= 0:
      num_annotations_skipped += 1
      continue
    if x + width > image_width or y + height > image_height:
      num_annotations_skipped += 1
      continue
    xmin.append(float(x) / image_width)
    xmax.append(float(x + width) / image_width)
    ymin.append(float(y) / image_height)
    ymax.append(float(y + height) / image_height)
    # is_crowd.append(object_annotations['iscrowd'])
    # category_id = int(object_annotations['category_id'])
    category_ids.append(category_id)
    category_names.append(category_index[category_id].encode('utf8'))
    area.append(width*height)

  feature_dict = {
      'image/height':
          dataset_util.int64_feature(image_height),
      'image/width':
          dataset_util.int64_feature(image_width),
      'image/filename':
          dataset_util.bytes_feature(filename.encode('utf8')),
      'image/source_id':
          dataset_util.bytes_feature(str(image_id).encode('utf8')),
      'image/key/sha256':
          dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded':
          dataset_util.bytes_feature(encoded_jpg),
      'image/format':
          dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin':
          dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax':
          dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin':
          dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax':
          dataset_util.float_list_feature(ymax),
      'image/object/class/text':
          dataset_util.bytes_list_feature(category_names),
      'image/object/area':
          dataset_util.float_list_feature(area),
  }
  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  return key, example, num_annotations_skipped


def _create_tf_record_from_youtube_annotations(
    video, image_dir, output_path, include_masks, frame_range, num_shards,
    coco_map_path='./mscoco_label_map.pbtxt'):
  """Loads COCO annotation json files and converts to tf.Record format.

  Args:
    annotations_file: JSON file containing bounding box annotations.
    image_dir: Directory containing the image files.
    output_path: Path to output tf.Record file.
    include_masks: Whether to include instance segmentations masks
      (PNG encoded) in the result. default: False.
    frame_range: Specify the frame idx range that should be included in the 
        output tf record.
    num_shards: number of output file shards.
  """
  # load annotations
  # key: int, frame index
  # value: [x, y, x+w, y+h, t, score, obj_id]
  annotations_index = video.get_video_detection()
  resol = video.resolution
  category_index = load_COCOlabelmap(coco_map_path)
  with contextlib2.ExitStack() as tf_record_close_stack:
    output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
        tf_record_close_stack, output_path, num_shards)
    

    # groundtruth_data = json.load(fid)
    # images = groundtruth_data['images']
    # category_index = label_map_util.create_category_index(
    #     groundtruth_data['categories'])

    # annotations_index = {}
    # if 'annotations' in groundtruth_data:
    #   tf.logging.info(
    #       'Found groundtruth annotations. Building annotations index.')
    #   for annotation in groundtruth_data['annotations']:
    #     image_id = annotation['image_id']
    #     if image_id not in annotations_index:
    #       annotations_index[image_id] = []
    #     annotations_index[image_id].append(annotation)
    missing_annotation_count = 0
    total_num_annotations_skipped = 0

    for index in range(frame_range[0], frame_range[1]):
      image = {}
      image['height'] = resol[1]
      image['width'] = resol[0]
      image['filename'] = '{0:06d}.jpg'.format(index)      
      image['id'] = '{0:06d}.jpg'.format(index) 
      # if index not in annotations_index:
      #   missing_annotation_count += 1
      #   annotations_index[index] = []
      assert index in annotations_index, print('frame {} not in annotation'.format(index))
      if index % 100 == 0:
        tf.logging.info('On image %d of %d', index, frame_range[1] - frame_range[0])
      annotations_list = annotations_index[index]
      _, tf_example, num_annotations_skipped = create_tf_example(
          image, annotations_list, image_dir, category_index, include_masks)
      total_num_annotations_skipped += num_annotations_skipped
      shard_idx = index % num_shards
      output_tfrecords[shard_idx].write(tf_example.SerializeToString())
    tf.logging.info('Finished writing, skipped %d annotations.',
                    total_num_annotations_skipped)



def main(_):
  """ do the train and val record file generation """
  required_flags = ['data_path', 
                    'output_path', 'dataset_name']

  for flag_name in required_flags:
      if not getattr(FLAGS, flag_name):
          raise ValueError('Flag --{} is required'.format(flag_name))
  
  if not os.path.exists(FLAGS.output_path):
      os.makedirs(FLAGS.output_path)

  train_output_path = os.path.join(FLAGS.output_path, FLAGS.dataset_name + '_train.record')
  val_output_path = os.path.join(FLAGS.output_path, FLAGS.dataset_name + '_val.record')

  train_range = [int(x) for x in FLAGS.train_range.split(',')]
  val_range = [int(x) for x in FLAGS.val_range.split(',')]
  print('training data range:', train_range)
  print('val data range:', val_range)


  metadata_file = FLAGS.data_path + '/{}/metadata.json'.format(FLAGS.dataset_name)
  dt_file = os.path.join(
      FLAGS.data_path, FLAGS.dataset_name, FLAGS.resol,
      'profile/updated_gt_FasterRCNN_COCO_no_filter.csv')
  img_path = os.path.join(
      FLAGS.data_path, FLAGS.dataset_name, FLAGS.resol)
  video = YoutubeVideo(FLAGS.dataset_name, FLAGS.resol, metadata_file, dt_file, img_path)

  _create_tf_record_from_youtube_annotations(
      video,
      img_path,
      train_output_path,
      FLAGS.include_masks,
      train_range,
      num_shards=10)
  _create_tf_record_from_youtube_annotations(
      video,
      img_path,
      val_output_path,
      FLAGS.include_masks,
      val_range,
      num_shards=1)



if __name__ == '__main__':
  tf.app.run()
