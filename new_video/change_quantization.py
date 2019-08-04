import cv2
import os
import subprocess
from absl import app, flags
from my_utils import load_metadata, create_dir

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', '', 'Dataset name')
flags.DEFINE_string('path', None, 'Data path')
flags.DEFINE_string('quality_parameter', 'original', 'Quality Level')
flags.DEFINE_string('resolution', 'original', 'Video Resolution')

def extract_frames(video, output_path):
    cmd = "ffmpeg -y -i {} {}%06d.jpg -hide_banner".format(video, output_path)
    print(cmd)
    os.system(cmd)


def main(argv):
    dataset = FLAGS.dataset
    path = FLAGS.path
    if FLAGS.quality_parameter == 'original':
        return
    qp = int(FLAGS.quality_parameter) 
    resolution = FLAGS.resolution
    assert qp >= 0 and qp <= 51
    
    metadata = load_metadata(path + dataset +'/metadata.json')
    frame_count = metadata['frame count']
    if resolution == 'original':
        img_path = path + dataset + '/'
        in_video = img_path + dataset + '.mp4'
        out_path = img_path + 'qp' + str(qp) + '/'
        out_video = out_path + dataset + '_qp' + str(qp) + '.mp4'
    else:
        img_path = path + dataset + '/' + resolution + '/'
        in_video = img_path + dataset + '_' + resolution + '.mp4'
        out_path = img_path + 'qp' + str(qp) + '/'
        out_video = out_path + dataset + '_' + resolution + '_qp' + str(qp) + '.mp4'

    create_dir(out_path)
    create_dir(out_path + '/profile')
    cmd = ['ffmpeg', '-n', '-i', in_video, '-vcodec', 'libx264', 
           '-qp', str(qp), '-hide_banner', out_video]
    subprocess.run(cmd)

    extract_frames(out_video, out_path)

if __name__ == '__main__':
    app.run(main)
