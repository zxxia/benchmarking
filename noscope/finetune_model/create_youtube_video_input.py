""" create input.record file  """
import os
import tensorflow as tf
from object_detection.utils import dataset_util
#  from collections import defaultdict
#  from PIL import Image
#  import PIL
#  import io
from benchmarking.utils.utils import load_metadata
from benchmarking.constants import RESOL_DICT

flags = tf.app.flags
flags.DEFINE_string('resol', 'original',
                    'Image resolution after resizing.')
flags.DEFINE_string('data_path', '', 'Data path.')
flags.DEFINE_string('metadata_file', '', 'Metadata file.')
flags.DEFINE_string('output_path', '', 'Dataset name.')
flags.DEFINE_string('start_frame', None, 'Start frame')
flags.DEFINE_string('end_frame', None, 'End fame(included).')

FLAGS = flags.FLAGS


def create_tf_example(image, image_dir, include_masks=False):
    """ do the input record file generation """
    # TODO(user): Populate the following variables from your example.

    filename = image['filename']
    # Filename of the image. Empty if image is not from file
    full_path = os.path.join(image_dir, filename)

    height = image['height']  # Image height
    width = image['width']  # Image width
    with tf.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()

    encoded_image_data = encoded_jpg  # Encoded image bytes
    image_format = b'jpg'  # b'jpeg' or b'png'
    #  num_annot_skipped = 0

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
    """ do the input record file generation """
    required_flags = ['metadata_file', 'data_path', 'output_path']

    for flag_name in required_flags:
        if not getattr(FLAGS, flag_name):
            raise ValueError('Flag --{} is required'.format(flag_name))
    data_path = FLAGS.data_path
    output_path = FLAGS.output_path
    output_file = output_path + 'input.record'
    print(output_path)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    writer = tf.python_io.TFRecordWriter(output_file)

    metadata = load_metadata(FLAGS.metadata_file)
    img_resolution = metadata['resolution']

    if FLAGS.resol:
        img_resolution = RESOL_DICT[FLAGS.resol]
    if FLAGS.start_frame is not None:
        start_frame = int(FLAGS.start_frame)
    else:
        start_frame = 1
    if FLAGS.end_frame is not None:
        end_frame = int(FLAGS.end_frame) + 1
    else:
        end_frame = metadata['frame count']+1

    for index in range(start_frame, end_frame):
        image = {}
        if FLAGS.resol != 'original':
            resol = RESOL_DICT[FLAGS.resol]
            image['height'] = resol[1]
            image['width'] = resol[0]
        else:
            image['height'] = img_resolution[1]
            image['width'] = img_resolution[0]
        image['filename'] = '{0:06d}.jpg'.format(index)
        tf_example = create_tf_example(image, data_path)
        writer.write(tf_example.SerializeToString())
    writer.close()


if __name__ == '__main__':
    tf.app.run()
