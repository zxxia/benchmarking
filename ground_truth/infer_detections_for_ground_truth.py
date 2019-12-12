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
import time
import os
import tensorflow as tf
import detection_inference_for_ground_truth as detection_inference
# from object_detection.metrics import tf_example_parser


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
tf.flags.DEFINE_string('gt_csv', None, 'Path to ground truth csv.')
tf.flags.DEFINE_boolean('discard_image_pixels', True,
                        'Discards the images in the output TFExamples. This'
                        ' significantly reduces the output size and is useful'
                        ' if the subsequent tools don\'t need access to the'
                        ' images (e.g. when computing evaluation measures).')

FLAGS = tf.flags.FLAGS


def main(_):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    all_time = []
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    required_flags = ['input_tfrecord_paths', 'output_tfrecord_path',
                      'inference_graph', 'output_time_path', 'gt_csv']

    for flag_name in required_flags:
        if not getattr(FLAGS, flag_name):
            raise ValueError('Flag --{} is required'.format(flag_name))

    f = open(FLAGS.output_time_path, 'w')
    gt_f = open(FLAGS.gt_csv, 'w')
    gt_f.write('image name, bounding boxes (x, y, w, h, type, score)\n')

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    with tf.compat.v1.Session(config=config) as sess:
        input_tfrecord_paths = [
            v for v in FLAGS.input_tfrecord_paths.split(',') if v]
        tf.compat.v1.logging.info('Reading input from %d files',
                        len(input_tfrecord_paths))
        serialized_example_tensor, image_tensor, image_filename_tensor, height, width \
            = detection_inference.build_input(input_tfrecord_paths)
        tf.compat.v1.logging.info('Reading graph and building model...')
        (detected_boxes_tensor, detected_scores_tensor, detected_labels_tensor) \
            = detection_inference.build_inference_graph(image_tensor, FLAGS.inference_graph)

        tf.compat.v1.logging.info('Running inference and writing output to {}'.format(
            FLAGS.output_tfrecord_path))
        sess.run(tf.compat.v1.local_variables_initializer())
        tf.train.start_queue_runners()
        with tf.io.TFRecordWriter(FLAGS.output_tfrecord_path) as tf_record_writer:
            try:
                for counter in itertools.count():
                    tf.compat.v1.logging.log_every_n(tf.compat.v1.logging.INFO,
                                           'Processed %d images...',
                                           10, counter)
                    start_time = time.time()
                    tf_example, image_filename = \
                        detection_inference. \
                        infer_detections_and_add_to_example(
                            gt_f, serialized_example_tensor,
                            detected_boxes_tensor, detected_scores_tensor,
                            detected_labels_tensor, image_filename_tensor,
                            height, width, FLAGS.discard_image_pixels)
                    all_time.append(time.time()-start_time)
                    image_filename = image_filename.decode("utf-8")
                    f.write(image_filename + ','
                            + "{:.9f}".format(time.time()-start_time) + '\n')
                    tf_record_writer.write(tf_example.SerializeToString())
            except tf.errors.OutOfRangeError:
                tf.compat.v1.logging.info('Finished processing records')
    print(sum(all_time)/len(all_time))


if __name__ == '__main__':
    tf.compat.v1.app.run()
