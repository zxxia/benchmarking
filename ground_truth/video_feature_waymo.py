
import sys
import imp
# sys.path.append('/Users/zhujunxiao/Desktop/benchmarking/code')
# gen_util = imp.load_source('utils', '/Users/zhujunxiao/Desktop/benchmarking/code/my_utils.py')
# from utils import IoU

from utils.utils import IoU, load_metadata
import glob
import csv
import os
import time
import numpy as np
from PIL import Image
from collections import defaultdict
from video_feature_KITTI import compute_para
from absl import app, flags
import re
import pdb

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'highway', 'The name of youtube video.')
flags.DEFINE_string('path', None, 'Data path.')
#flags.DEFINE_integer('frame_rate', 30, 'Frame rate.')
#flags.DEFINE_string('resol', None, 'Video resolution.')


PATH = '/mnt/data/zhujun/new_video/'
DATASET_LIST = ['training_0000', 'training_0001', 'training_0002',
                'validation_0000']


def read_annot(annot_path, target_object=[1]):
    all_filename = []
    frame_to_object = defaultdict(list)
    object_to_frame = defaultdict(list)
    object_location = {}

    with open(annot_path, 'r') as f:
        # f.readline()
        for line in f:
            # each line: (frame_id, object_id, x, y, w, h, object_type)
            line_list = line.strip().split(',')
            frame_id = int(line_list[0])
            object_id = line_list[1]
            if int(float(line_list[6])) not in target_object:
                continue
            frame_to_object[frame_id].append(line_list[1:])
            object_to_frame[object_id].append(frame_id)
            all_filename.append(frame_id)
            key = (frame_id, object_id)
            [x, y, w, h] = [int(float(x)) for x in line_list[2:6]]
            object_location[key] = [x, y, w, h]
    return all_filename, frame_to_object, object_to_frame, object_location


def main(argv):
    video_name = 'training_0000'  # FLAGS.dataset
    path = PATH

    metadata = load_metadata(path + 'metadata.json')
    frame_rate = metadata['frame rate']
    image_resolution = metadata['resolution']
    for dataset in DATASET_LIST:
        segments = sorted(glob.glob(PATH + dataset + '/*.tfrecord'))
        for segment_file in segments:
            rz = re.findall("/(segment.*)_with_camera_labels.tfrecord",
                            segment_file)
            segment_name = rz[0]
            current_path = path + 'waymo_groundtruths/' + dataset + '/'
            annot_path = current_path + 'Parsed_gt_{}.csv'.format(segment_name)
            print(annot_path)

            # read annotations from ground truth file
            data = read_annot(annot_path)
            paras = compute_para(data, image_resolution, frame_rate)

            all_filename = data[0]
            current_start = min(all_filename)
            current_end = max(all_filename)
            para_file = current_path + 'Video_features_' + segment_name + '.csv'
            with open(para_file, 'w') as f:
                f.write('frame_id, num_of_object, object_area, arrival_rate,'
                        'velocity, total_object_area, num_of_object_type, dominate_object_type\n')
                for frame_id in range(current_start, current_end + 1 - frame_rate):
                    f.write(str(frame_id) + ',' +
                            str(paras.num_of_objects[frame_id]) + ',' +
                            str(paras.object_area[frame_id]) + ',' +
                            str(paras.arrival_rate[frame_id]) + ',' +
                            str(paras.velocity[frame_id]) + ',' +
                            str(paras.total_object_area[frame_id]) + ',' +
                            str(paras.object_type[frame_id]) + ',' +
                            str(paras.dominate_object_type[frame_id]) + '\n')


if __name__ == '__main__':
    app.run(main)
