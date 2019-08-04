import tensorflow as tf
import os
from object_detection.utils import dataset_util
from collections import defaultdict
from PIL import Image
import PIL
import io
from my_utils import load_metadata

flags = tf.app.flags
flags.DEFINE_string('dataset', '', 'Dataset name.')
# flags.DEFINE_integer('frame_count', 0, 'Total number of frames.')
# flags.DEFINE_string('resol','','Image resolution.')
#flags.DEFINE_boolean('resize', None, 'Resize the image or not.')
flags.DEFINE_string('resize_resol','original','Image resolution after resizing.')
flags.DEFINE_string('path', None, 'Data path.')
flags.DEFINE_string('quality_parameter', 'original', 'Quality Parameter')

FLAGS = flags.FLAGS


resol_dict = {'360p': [640,360],
              '480p': [854, 480],
              '540p': [960, 540],
              '576p': [1024, 576],
              '720p': [1280, 720],}


def create_tf_example(image, image_dir, include_masks=False):
  # TODO(user): Populate the following variables from your example.

  filename = image['filename'] # Filename of the image. Empty if image is not from file
  full_path = os.path.join(image_dir, filename)

  if FLAGS.resize_resol != 'original':
    resize_resol = resol_dict[FLAGS.resize_resol]
    height = resize_resol[1]
    width = resize_resol[0]
  else:
    height = image['height'] # Image height
    width = image['width'] # Image width
  with tf.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()



  encoded_image_data = encoded_jpg # Encoded image bytes
  image_format = b'jpg' # b'jpeg' or b'png'
  num_annot_skipped = 0

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format)
  }))
  return tf_example


def main(_):
  path =  FLAGS.path
  assert FLAGS.dataset is not None
  output_path = path + FLAGS.dataset + '/profile/'

  metadata = load_metadata(path + FLAGS.dataset + '/metadata.json')

  if FLAGS.resize_resol != 'original':
    if FLAGS.quality_parameter != 'original':
      data_path = path + FLAGS.dataset  + '/' + FLAGS.resize_resol + '/' + \
                  'qp' + FLAGS.quality_parameter + '/'
    else:
      data_path = path + FLAGS.dataset  + '/' + FLAGS.resize_resol + '/'
    output_file = data_path + '/profile/input_' + FLAGS.resize_resol + '.record'
  else:
    if FLAGS.quality_parameter != 'original':
      data_path = path + FLAGS.dataset + '/qp' + FLAGS.quality_parameter + '/' 
    else:
      data_path = path + FLAGS.dataset 
    output_file = data_path + '/profile/input.record'

  if not os.path.exists(output_path):
    os.mkdir(output_path)

  writer = tf.python_io.TFRecordWriter(output_file)

  img_resolution = metadata['resolution']


  for index in range(1, 9000):#metadata['frame count'] + 1):
    image = {}
    image['height'] = img_resolution[1]
    image['width'] = img_resolution[0]
    image['filename'] = '{0:06d}.jpg'.format(index)
    tf_example = create_tf_example(image, data_path)
    writer.write(tf_example.SerializeToString())
  writer.close()
 


if __name__ == '__main__':
  tf.app.run()
