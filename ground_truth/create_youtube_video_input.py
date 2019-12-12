"""Create input.record file."""
import os
import glob
import tensorflow as tf
from object_detection.utils import dataset_util
from PIL import Image

flags = tf.app.flags
flags.DEFINE_string('data_path', '', 'Data path.')
flags.DEFINE_string('output_path', '', 'Dataset name.')
flags.DEFINE_string('dataset', None, 'dataset name.')

FLAGS = flags.FLAGS


def create_tf_example(image, image_dir, include_masks=False):
    """Do the input record file generation."""
    # TODO(user): Populate the following variables from your example.

    filename = image['filename']
    # Filename of the image. Empty if image is not from file
    full_path = os.path.join(image_dir, filename)

    height = image['height']  # Image height
    width = image['width']  # Image width
    with tf.io.gfile.GFile(full_path, 'rb') as fid:
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
    """Do the input record file generation."""
    required_flags = ['data_path', 'output_path']

    for flag_name in required_flags:
        if not getattr(FLAGS, flag_name):
            raise ValueError('Flag --{} is required'.format(flag_name))
    data_path = FLAGS.data_path
    output_path = FLAGS.output_path
    output_file = os.path.join(output_path, 'input.record')

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    writer = tf.io.TFRecordWriter(output_file)

    if FLAGS.dataset == 'kitti':  # is not None:
        img_name_template = '*.png'
        # image['filename'] = '{0:010d}.png'.format(index)
    elif FLAGS.dataset == 'waymo':
        img_name_template = '*.jpg'
        # image['filename'] = '{0:04d}.jpg'.format(index)
    else:
        img_name_template = '*.jpg'
        # image['filename'] = '{0:06d}.jpg'.format(index)
    img_paths = sorted(glob.glob(os.path.join(data_path, img_name_template)))

    for img_path in img_paths:
        image = {}
        img = Image.open(img_path)
        image['filename'] = os.path.basename(img_path)
        image['width'] = img.size[0]
        image['height'] = img.size[1]
        img.close()
        tf_example = create_tf_example(image, data_path)
        writer.write(tf_example.SerializeToString())
    writer.close()


if __name__ == '__main__':
    tf.compat.v1.app.run()
