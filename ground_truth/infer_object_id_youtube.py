import sys
import imp
import glob
import csv
import os
import time
import numpy as np
from PIL import Image
from absl import app, flags
from collections import defaultdict, Counter
from utils.utils import nms, Most_Common, load_metadata
# from show_annot import show
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', None, 'The name of youtube video.')
flags.DEFINE_string('path', None, 'Data path.')
# flags.DEFINE_string('resol', None, 'Video resolution.')
# flags.DEFINE_boolean('resize', None, 'Resize the image or not.')
flags.DEFINE_string('resize_resol', 'original', 'Image resolution after resizing.')
flags.DEFINE_string('quality_parameter', 'original', 'Quality parameter')


def tag_object(all_filename, frame_to_object, output_folder):
    output_file = output_folder + 'Parsed_gt_FasterRCNN_COCO.csv'
    update_gt_file = output_folder + 'updated_gt_FasterRCNN_COCO.csv'
    gt_f = open(update_gt_file, 'w')
    last_frame = []
    object_to_type = defaultdict(list)
    object_cn = 0
    object_id = 0
    new_frame_to_object = defaultdict(list)
    # thresh = 12 # distance should not be larger than 12 pixel
    with open(output_file, 'w') as f:
        f.write('Frame ID, Object ID, X, Y, W, H, Object Type, Score\n')
        for filename in all_filename:
            boxes = frame_to_object[filename]
            current_frame = []
            for box in boxes:
                # import pdb
                # pdb.set_trace()
                assert len(box) == 6, print(filename, boxes)
                # find the center point of the bounding box
                x_c = box[0] + box[2]/2.0
                y_c = box[1] + box[3]/2.0
                thresh = 0.45 * box[3]
                dist_with_last_frame = []
                if not last_frame:
                    object_cn += 1
                    object_id = object_cn
                else:
                    for last_box in last_frame:
                        dist = np.linalg.norm(np.array([x_c, y_c])
                                              - np.array([last_box[0],
                                                          last_box[1]]))
                        dist_with_last_frame.append(dist)
                    min_dist = min(dist_with_last_frame)
                    index = dist_with_last_frame.index(min_dist)
                    if min_dist > thresh:
                        # new object
                        object_cn += 1
                        object_id = object_cn
                    else:
                        # old object, get the id
                        object_id = last_frame[index][2]
                        last_frame.remove(last_frame[index])
                object_to_type[object_id].append(box[4])
                # box.append(object_id)
                new_frame_to_object[filename].append(box+[object_id])
                current_frame.append([x_c, y_c, object_id])

            last_frame = current_frame
        # update the type for each object
        for object_id in object_to_type.keys():
            types = object_to_type[object_id]
            object_to_type[object_id] = Most_Common(types)

        frame_to_object = defaultdict(list)
        for filename in all_filename:
            # import pdb
            # pdb.set_trace()
            boxes = new_frame_to_object[filename]
            for box in boxes:
                assert len(box) == 7, print(box)
                object_id = box[-1]
                box[4] = object_to_type[object_id]
                f.write(str(filename) + ',' + str(object_id) + ',' +
                        ','.join(str(x) for x in box[0:6]) + '\n')

                frame_to_object[filename].append(box)
            # show(filename, path + 'crossroad2/images/', frame_to_object)
            gt_f.write(str(filename) + ',' + ';'
                       .join([' '.join([str(x) for x in box])
                             for box in frame_to_object[filename]]) + '\n')

    return


def smooth_annot(all_filename, frame_to_object):
    # for each frame, if a box exists in last frame and next frame
    # but not in current frame, add it to current frame
    # update_gt_file = output_folder + 'updated_gt_FasterRCNN_COCO.csv'

    # with open(update_gt_file, 'w') as f:

    for filename in all_filename:
        if not frame_to_object[filename - 1] or not frame_to_object[filename + 1]:
            # current_boxes = frame_to_object[filename]
            current_boxes = []
            # f.write(str(filename) + ',' + ';'.join([' '.join([str(x) for x in box])
            #                               for box in current_boxes])+ '\n')
        else:
            current_boxes = frame_to_object[filename]
            last_boxes = frame_to_object[filename-1]
            next_boxes = frame_to_object[filename+1]
            assert len(last_boxes) and len(next_boxes), print(filename)
            for box in next_boxes:
                x_c = box[0] + box[2]/2.0
                y_c = box[1] + box[3]/2.0
                thresh = 0.45 * box[3]
                dist_with_last_frame = []
                dist_with_current_frame = []
                # check if this box exists in last frame
                for last_box in last_boxes:
                    last_x_c = last_box[0] + last_box[2]/2.0
                    last_y_c = last_box[1] + last_box[3]/2.0
                    dist = np.linalg.norm(np.array([x_c, y_c]) -
                                          np.array([last_x_c, last_y_c]))
                    dist_with_last_frame.append(dist)
                min_last_dist = min(dist_with_last_frame)

                # check if this box exists in current frame
                if not current_boxes:
                    min_current_dist = 100
                else:
                    for current_box in current_boxes:
                        current_x_c = current_box[0] + current_box[2]/2.0
                        current_y_c = current_box[1] + current_box[3]/2.0
                        dist = np.linalg.norm(np.array([x_c, y_c]) -
                                              np.array([current_x_c,
                                                        current_y_c]))
                        dist_with_current_frame.append(dist)
                    min_current_dist = min(dist_with_current_frame)

                # if this box exist in last frame, but not in current frame
                # add this box to current frame
                if min_last_dist < thresh and min_current_dist > thresh:
                    assert len(box) == 6, print(box)
                    current_boxes.append(box)

            # if a box in current frame, but not in previous or next frame
            # remove this box in current frame
            for box in current_boxes:
                x_c = box[0] + box[2]/2.0
                y_c = box[1] + box[3]/2.0
                thresh = 0.45 * box[3]
                dist_with_last_frame = []
                dist_with_next_frame = []
                # check if this box exists in last frame
                for last_box in last_boxes:
                    last_x_c = last_box[0] + last_box[2]/2.0
                    last_y_c = last_box[1] + last_box[3]/2.0
                    dist = np.linalg.norm(np.array([x_c, y_c])
                                          - np.array([last_x_c, last_y_c]))
                    dist_with_last_frame.append(dist)
                min_last_dist = min(dist_with_last_frame)

                # check if this box exists in current frame

                for next_box in next_boxes:
                    next_x_c = next_box[0] + next_box[2]/2.0
                    next_y_c = next_box[1] + next_box[3]/2.0
                    dist = np.linalg.norm(np.array([x_c, y_c])
                                          - np.array([next_x_c, next_y_c]))
                    dist_with_next_frame.append(dist)

                min_next_dist = min(dist_with_next_frame)

                # if this box exist in last frame, but not in current frame
                # add this box to current frame
                if min_last_dist > thresh and min_next_dist > thresh:
                    assert len(box) == 6, print(box)
                    current_boxes.remove(box)

                frame_to_object[filename] = current_boxes
                # f.write(str(filename) + ',' + ';'.join([' '.join([str(x) for x in box])
                #               for box in frame_to_object[filename]]) + '\n')

    return frame_to_object


def read_annot(annot_path, height_thresh):
    all_filename = []
    frame_to_object = defaultdict(list)
    tmp = defaultdict(list)

    with open(annot_path, 'r') as f:
        f.readline()
        for line in f:
            # each line: (image_name, bounding boxes (x, y, w, h, object_type, score))
            line_list = line.strip().split(',')
            frame_id = int(line_list[0].replace('.jpg', ''))
            boxes_list = line_list[1].split(';')
            all_filename.append(frame_id)

            if line_list[1] == '':
                continue

            dets = np.empty((len(boxes_list), 5))
            for index in range(len(boxes_list)):
                box_list = boxes_list[index].split(' ')
                x = int(box_list[0])
                y = int(box_list[1])
                w = int(box_list[2])
                h = int(box_list[3])
                t = int(box_list[4])
                score = float(box_list[5])
                dets[index, 0] = x
                dets[index, 1] = y
                dets[index, 2] = x + w
                dets[index, 3] = y + h
                dets[index, 4] = score  # forget to output score, use 1 for every box. Fix this later
                tmp[frame_id].append([x, y, w, h, t, score])
            keep = nms(dets, 0.4)
            for index in keep:
                # if t == 1: # object is 'car'
                box_list = boxes_list[index].split(' ')
                x = int(box_list[0])
                y = int(box_list[1])
                w = int(box_list[2])
                h = int(box_list[3])
                t = int(box_list[4])
                score = float(box_list[5])

                if h >= height_thresh:
                    frame_to_object[frame_id].append([x, y, w, h, t, score])
    return all_filename, frame_to_object


def main(argv):
    resol_dict = {'360p': [640, 360],
                  '480p': [854, 480],
                  '540p': [960, 540],
                  '576p': [1024, 576],
                  '720p': [1280, 720],
                  '1080p': [1920, 1080],
                  '2160p': [3840, 2160]}

    path = FLAGS.path

    video_index = FLAGS.dataset

    metadata = load_metadata(path + video_index + '/metadata.json')

    frame_rate = metadata['frame rate']

    if FLAGS.resize_resol != 'original':
        image_resolution = resol_dict[FLAGS.resize_resol]
        if FLAGS.quality_parameter != 'original':
            annot_path = path + video_index + '/' + FLAGS.resize_resol + '/qp' \
                         + FLAGS.quality_parameter + '/profile/'
        else:
            annot_path = path + video_index + '/' + FLAGS.resize_resol + '/profile/'
        annot_file = annot_path + 'gt_FasterRCNN_COCO_' + FLAGS.resize_resol + '.csv'
    else:
        image_resolution = metadata['resolution']  # [int(x) for x in FLAGS.resol.split(',')]#image_resolution_dict[video_index]
        if FLAGS.quality_parameter != 'original':
            annot_path = path + video_index + '/qp' + FLAGS.quality_parameter \
                          + '/profile/'
        else:
            annot_path = path + video_index + '/profile/'
        annot_file = annot_path + 'gt_FasterRCNN_COCO.csv'
    print(annot_path)

    height_thresh = image_resolution[1]//20  # remove objects that are too small
    all_filename, frame_to_object = read_annot(annot_file, height_thresh)
    print('Done loading annot.')
    new_frame_to_object = smooth_annot(all_filename, frame_to_object)

    print('Done smoothing annot.')
    tag_object(all_filename, new_frame_to_object, annot_path)


if __name__ == '__main__':
    app.run(main)
