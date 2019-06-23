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
r"""Infers detections on a TFRecord of TFExamples given an inference graph.

Example usage:
  ./infer_detections \
    --input_tfrecord_paths=/path/to/input/tfrecord1,/path/to/input/tfrecord2 \
    --output_tfrecord_path_prefix=/path/to/output/detections.tfrecord \
    --inference_graph=/path/to/frozen_weights_inference_graph.pb

The output is a TFRecord of TFExamples. Each TFExample from the input is first
augmented with detections from the inference graph and then copied to the
output.

The input and output nodes of the inference graph are expected to have the same
types, shapes, and semantics, as the input and output nodes of graphs produced
by export_inference_graph.py, when run with --input_type=image_tensor.

The script can also discard the image pixels in the output. This greatly
reduces the output size and can potentially accelerate reading data in
subsequent processing steps that don't require the images (e.g. computing
metrics).
"""

import itertools
import tensorflow as tf
from object_detection.inference import detection_inference_for_ground_truth as detection_inference
from object_detection.metrics import tf_example_parser

import time
import os

tf.flags.DEFINE_string('gpu', None,
                       'GPU number.')
tf.flags.DEFINE_string('input_tfrecord_paths', None,
                       'A comma separated list of paths to input TFRecords.')
tf.flags.DEFINE_string('output_time_path', None,
                       'Path to the output GPU processing time file.')
tf.flags.DEFINE_string('output_tfrecord_path', None,
                       'Path to the output TFRecord.')
tf.flags.DEFINE_string('inference_graph', None,
                       'Path to the inference graph with embedded weights.')
tf.flags.DEFINE_string('gt_csv', None,
                        'Path to ground truth csv.')
tf.flags.DEFINE_boolean('discard_image_pixels', True,
                        'Discards the images in the output TFExamples. This'
                        ' significantly reduces the output size and is useful'
                        ' if the subsequent tools don\'t need access to the'
                        ' images (e.g. when computing evaluation measures).')
tf.flags.DEFINE_string('dataset',None,
                        'Dataset name')
tf.flags.DEFINE_boolean('resize', None, 'Resize the image or not.')
tf.flags.DEFINE_string('path', None, 'Data path.')
tf.flags.DEFINE_string('resize_resol',None,'Image resolution after resizing.')


FLAGS = tf.flags.FLAGS



def main(_):
  os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
  os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.gpu
  all_time = []
  tf.logging.set_verbosity(tf.logging.INFO)
  # data_path = '/home/zhujun/video_analytics_pipelines/dataset/Youtube/' + \
  #              FLAGS.dataset + '/profile/'
  data_path = FLAGS.path + FLAGS.dataset + '/profile/'
  print(data_path)
  if FLAGS.resize:
    data_path = FLAGS.path + FLAGS.dataset + '/' + FLAGS.resize_resol + '/profile/'
    FLAGS.input_tfrecord_paths = data_path + 'input_' + FLAGS.resize_resol + '.record'
    FLAGS.output_tfrecord_path = data_path + 'gt_FasterRCNN_COCO_' + \
                      FLAGS.resize_resol + '.record'
    FLAGS.output_time_path = data_path + 'full_model_time_FasterRCNN_COCO' + \
                      FLAGS.resize_resol + '.csv'
    FLAGS.gt_csv = data_path + 'gt_FasterRCNN_COCO_'+ \
                      FLAGS.resize_resol + '.csv'
  else:
    FLAGS.input_tfrecord_paths = data_path + 'input.record'
    FLAGS.output_tfrecord_path = data_path + 'gt_FasterRCNN_COCO.record'
    FLAGS.output_time_path = data_path + 'full_model_time_FasterRCNN_COCO.csv'
    FLAGS.gt_csv = data_path + 'gt_FasterRCNN_COCO.csv' 

  required_flags = ['input_tfrecord_paths', 'output_tfrecord_path',
                    'inference_graph', 'output_time_path','gt_csv']     

  for flag_name in required_flags:
    if not getattr(FLAGS, flag_name):
      raise ValueError('Flag --{} is required'.format(flag_name))


  f = open(FLAGS.output_time_path,'w')
  gt_f = open(FLAGS.gt_csv, 'w')
  gt_f.write('image name, bounding boxes (x, y, w, h, type)\n')

  config = tf.ConfigProto()
  config.gpu_options.per_process_gpu_memory_fraction = 0.4
  with tf.Session(config=config) as sess:
    input_tfrecord_paths = [
        v for v in FLAGS.input_tfrecord_paths.split(',') if v]
    tf.logging.info('Reading input from %d files', len(input_tfrecord_paths))
    serialized_example_tensor, image_tensor, image_filename_tensor, height, width \
        = detection_inference.build_input(input_tfrecord_paths)
    tf.logging.info('Reading graph and building model...')
    (detected_boxes_tensor, detected_scores_tensor,detected_labels_tensor) \
        = detection_inference.build_inference_graph(image_tensor, FLAGS.inference_graph)

    tf.logging.info('Running inference and writing output to {}'.format(
        FLAGS.output_tfrecord_path))
    sess.run(tf.local_variables_initializer())
    tf.train.start_queue_runners()
    with tf.python_io.TFRecordWriter(
        FLAGS.output_tfrecord_path) as tf_record_writer:
      try:
        for counter in itertools.count():
          tf.logging.log_every_n(tf.logging.INFO, 'Processed %d images...', 10,
                                 counter)
          start_time = time.time()
          tf_example, image_filename = detection_inference.infer_detections_and_add_to_example(
              gt_f, serialized_example_tensor, detected_boxes_tensor,
              detected_scores_tensor, detected_labels_tensor, 
              image_filename_tensor, height, width, FLAGS.discard_image_pixels)
          all_time.append(time.time()-start_time)
          image_filename = image_filename.decode("utf-8")
          f.write(image_filename + ',' + "{:.9f}".format(time.time()-start_time) + '\n')
          tf_record_writer.write(tf_example.SerializeToString())
      except tf.errors.OutOfRangeError:
        tf.logging.info('Finished processing records')
  print(sum(all_time)/len(all_time ))


if __name__ == '__main__':
  tf.app.run()
