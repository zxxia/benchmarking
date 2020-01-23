""" To smooth bounding boxes and assign object ID """
# import sys
# import imp
# import glob
# import csv
import os
# import time
import pdb
import copy
from collections import defaultdict  # , Counter
import numpy as np
# from PIL import Image
from absl import app, flags
import sys
from benchmarking.utils.utils import nms, Most_Common, IoU
# from show_annot import show
FLAGS = flags.FLAGS
flags.DEFINE_string('resol', 'None', 'Image resolution.')
#  flags.DEFINE_string('quality_parameter', 'original', 'Quality parameter')
# flags.DEFINE_string('metadata_file', '', 'metadata file')
flags.DEFINE_string('input_file', None, 'input file')
flags.DEFINE_string('output_file', None, 'output file')
# flags.DEFINE_string('updated_gt_file', None, 'updated gt file')
flags.DEFINE_string('model_name', 'FasterRCNN', '')
# flags.DEFINE_string('overlap_percent', '', 'updated gt file')
# flags.DEFINE_string('vote_percent', '', 'updated gt file')
# flags.DEFINE_string('win_size', '', 'updated gt file')


def tag_object(all_filename, frame_to_object, update_gt_file,  # output_file,
               dist_coef=0.45):
    """Assign object id to bounding boxes.

    Args
        frame_to_object: a dict in which frame id maps to object detections
        output_file and update_gt_file stores the detections with object id
        but in different format
        dist_coef: adjust the threshold of distance which is used to determine
        whether two bounding boxes belong to the same object, if use FasterRCNN
        dist_coef should be set to 0.45. Otherwise, mobilenet needs to be 0.8

    """
    last_frame = []
    object_to_type = defaultdict(list)
    object_cn = 0
    object_id = 0
    new_frame_to_object = defaultdict(list)
    # thresh = 12 # distance should not be larger than 12 pixel
    # with open(output_file, 'w') as f:
    #     f.write('Frame ID, Object ID, X, Y, W, H, Object Type, Score\n')
    with open(update_gt_file, 'w') as gt_f:
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
                thresh = dist_coef * box[3]
                dist_with_last_frame = []
                if not last_frame:
                    object_cn += 1
                    object_id = object_cn
                else:
                    for last_box in last_frame:
                        dist = np.linalg.norm(
                            np.array([x_c, y_c]) - np.array([last_box[0],
                                                             last_box[1]]))
                        dist_with_last_frame.append(dist)
                    min_dist = min(dist_with_last_frame)
                    index = dist_with_last_frame.index(min_dist)
                    if min_dist > thresh:
                        # new object
                        object_cn += 1
                        object_id = object_cn
                        # print('dist is large:', filename, min_dist,
                        #       thresh, object_id)
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
            boxes = new_frame_to_object[filename]
            for box in boxes:
                assert len(box) == 7, print(box)
                object_id = box[-1]
                box[4] = object_to_type[object_id]
                # f.write(str(filename) + ',' + str(object_id) + ',' +
                #         ','.join(str(x) for x in box[0:6]) + '\n')

                frame_to_object[filename].append(box)
            # show(filename, path + 'crossroad2/images/', frame_to_object)
            gt_f.write(str(filename) + ',' + ';'
                       .join([' '.join([str(x) for x in box])
                              for box in frame_to_object[filename]]) + '\n')


def smooth_annot(all_filename, frame_to_object):
    """for each frame, if a box exists in last frame and next frame
    but not in current frame, add it to current frame
    update_gt_file = output_folder + 'updated_gt_FasterRCNN_COCO.csv'
    """

    # with open(update_gt_file, 'w') as f:

    for filename in all_filename:
        if not frame_to_object[filename-1] or not frame_to_object[filename+1]:
            # current_boxes = frame_to_object[filename]
            current_boxes = []
            # f.write(str(filename)+','+';'.join([' '
            #                                     .join([str(x) for x in box])
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
                    dist = np.linalg.norm(
                        np.array([x_c, y_c]) - np.array([last_x_c, last_y_c]))
                    dist_with_last_frame.append(dist)
                min_last_dist = min(dist_with_last_frame)

                # check if this box exists in current frame

                for next_box in next_boxes:
                    next_x_c = next_box[0] + next_box[2]/2.0
                    next_y_c = next_box[1] + next_box[3]/2.0
                    dist = np.linalg.norm(
                        np.array([x_c, y_c]) - np.array([next_x_c, next_y_c]))
                    dist_with_next_frame.append(dist)

                min_next_dist = min(dist_with_next_frame)

                # if this box exist in last frame, but not in current frame
                # add this box to current frame
                if min_last_dist > thresh and min_next_dist > thresh:
                    assert len(box) == 6, print(box)
                    current_boxes.remove(box)

                frame_to_object[filename] = current_boxes

    return frame_to_object


def smooth(input_frame_to_obj, overlap_percent=0.2,
           vote_percent=0.2, win_size=10):
    """Smooth the bounding boxes.

    Args:
        input_frame_to_obj: a dict which maps frame id to a list of
                            bounding boxes
        overlap_percent: the percentage threshold of iou. when above this
                         number, two boxes are considered to belong to the
                         same object
        win_size: the window size which smoothing will be applied to

    """
    frame_indices = sorted(input_frame_to_obj.keys())
    min_frame_idx = min(frame_indices)
    max_frame_idx = max(frame_indices)
    tagged = defaultdict(list)

    frame_to_obj = copy.deepcopy(input_frame_to_obj)

    # for frame_idx in np.arange(min_frame_idx, 51, 10):
    for frame_idx in np.arange(min_frame_idx, max_frame_idx+1, 10):
        frame_start = frame_idx
        frame_end = frame_idx + win_size
        # within window
        # print('win = [{}, {}]'.format(frame_start, frame_end))
        for i in range(frame_start, frame_end):
            boxes_i = frame_to_obj[i]
            # if i == 1:
            #     print(boxes_i)
            votes = np.zeros((len(boxes_i), win_size))
            ious = np.zeros((len(boxes_i), win_size))
            for obj_idx_i, box_i in enumerate(boxes_i):
                obj_id_record = [-1] * win_size
                obj_id_record[i-frame_start] = obj_idx_i
                if obj_idx_i in tagged[i]:
                    continue
                tagged[i].append(obj_idx_i)
                votes[obj_idx_i, i-frame_start] = 1
                # for j in range(i+1, frame_end):
                for j in range(frame_start, frame_end):
                    boxes_j = frame_to_obj[j]

                    for obj_idx_j, box_j in enumerate(boxes_j):
                        tmp_box_i = copy.deepcopy(box_i)
                        tmp_box_j = copy.deepcopy(box_j)
                        tmp_box_i[2] += tmp_box_i[0]
                        tmp_box_i[3] += tmp_box_i[1]
                        tmp_box_j[2] += tmp_box_j[0]
                        tmp_box_j[3] += tmp_box_j[1]
                        iou = IoU(tmp_box_i, tmp_box_j)

                        ious[obj_idx_i, j-frame_start] = iou
                        if obj_idx_j not in tagged[j] and \
                           iou > overlap_percent:
                            votes[obj_idx_i, j-frame_start] = 1
                            tagged[j].append(obj_idx_j)
                            obj_id_record[j-frame_start] = obj_idx_j
                            # mark this box is taged

                # print(votes[obj_idx_i])
                # print(ious[obj_idx_i])
                if np.sum(votes[obj_idx_i])/win_size >= vote_percent:
                    # TODO: interpolation

                    # Forward filling up
                    # for j in range(i+1, frame_end):
                    for j in range(frame_start, frame_end):
                        if votes[obj_idx_i, j-frame_start] == 0:
                            # tag the filed objects
                            # print('before', len(frame_to_obj[j]))

                            # fill up missing objects based on with the closest
                            # upvotes
                            left_steps = 0
                            right_steps = 0
                            for ri in range(j-frame_start, win_size):
                                if votes[obj_idx_i, ri]:
                                    break
                                right_steps += 1
                            for li in range(j-frame_start, -1, -1):
                                if votes[obj_idx_i, li]:
                                    break
                                left_steps -= 1
                            if votes[obj_idx_i, ri] and votes[obj_idx_i, li]:
                                if abs(left_steps) >= right_steps:
                                    steps = right_steps
                                else:
                                    steps = left_steps
                            elif votes[obj_idx_i, li]:
                                steps = left_steps
                            elif votes[obj_idx_i, ri]:
                                steps = right_steps
                            else:
                                continue

                            # frame_to_obj[j].append(box_i)
                            try:
                                to_be_add_box = \
                                    frame_to_obj[j +
                                                 steps][obj_id_record[j-frame_start+steps]]
                                add_flag = True
                                for box in frame_to_obj[j]:
                                    tmp_box = box.copy()
                                    tmp_to_be_add_box = to_be_add_box.copy()

                                    tmp_box[2] += tmp_box[0]
                                    tmp_box[3] += tmp_box[1]
                                    tmp_to_be_add_box[2] += tmp_to_be_add_box[0]
                                    tmp_to_be_add_box[3] += tmp_to_be_add_box[1]
                                    if IoU(tmp_box, tmp_to_be_add_box) > 0.4:
                                        add_flag = False
                                if add_flag:
                                    tagged[j].append(len(frame_to_obj[j]))
                                    frame_to_obj[j].append(
                                        frame_to_obj[j+steps][obj_id_record[j-frame_start+steps]])
                                    votes[obj_idx_i, j-frame_start] = 1
                            except IndexError:
                                print(votes[obj_idx_i], j-frame_start)
                                print(j-frame_start+steps, len(obj_id_record))
                                pdb.set_trace()

                            # print('after', len(frame_to_obj[j]))
                            # print('add {} from frame {} to frame {}'
                            #       .format(box_i, i, j))
    return frame_to_obj


def read_annot(annot_path):
    all_filename = []
    frame_to_object = defaultdict(list)
    tmp = defaultdict(list)

    with open(annot_path, 'r') as f:
        f.readline()
        for line in f:
            # each line:
            # (image_name, bounding boxes (x, y, w, h, object_type, score))
            line_list = line.strip().split(',')
            frame_id = int(os.path.splitext(line_list[0])[0])
            boxes_list_old = line_list[1].split(';')
            all_filename.append(frame_id)

            if line_list[1] == '':
                continue

            boxes_list = []
            for index in range(len(boxes_list_old)):
                box_list = boxes_list_old[index].split(' ')
                assert len(box_list) >=6, print(boxes_list_old)
                score = float(box_list[5])
                if score < 0.3:
                    continue
                boxes_list.append(boxes_list_old[index])

            dets = np.empty((len(boxes_list), 5))
            for index in range(len(boxes_list)):
                box_list = boxes_list[index].split(' ')
                if len(box_list) == 6:
                    x = int(box_list[0])
                    y = int(box_list[1])
                    w = int(box_list[2])
                    h = int(box_list[3])
                    t = int(box_list[4])
                    score = float(box_list[5])
                else:
                    x = int(box_list[0])
                    y = int(box_list[1])
                    w = int(box_list[2])
                    h = int(box_list[3])
                    t = int(box_list[4])
                    score = 1

                dets[index, 0] = x
                dets[index, 1] = y
                dets[index, 2] = x + w
                dets[index, 3] = y + h
                dets[index, 4] = score
                tmp[frame_id].append([x, y, w, h, t, score])
            keep = nms(dets, 0.4)
            for index in keep:
                box_list = boxes_list[index].split(' ')
                if len(box_list) == 6:
                    x = int(box_list[0])
                    y = int(box_list[1])
                    w = int(box_list[2])
                    h = int(box_list[3])
                    t = int(box_list[4])
                    score = float(box_list[5])
                else:
                    x = int(box_list[0])
                    y = int(box_list[1])
                    w = int(box_list[2])
                    h = int(box_list[3])
                    t = int(box_list[4])
                    score = 1
                # No filter applied in this stage
                frame_to_object[frame_id].append([x, y, w, h, t, score])
    return all_filename, frame_to_object


def main(argv):
    """ smooth bounding boxes and assign object id to bounding boxes """

    required_flags = ['input_file', 'output_file']

    for flag_name in required_flags:
        if not getattr(FLAGS, flag_name):
            raise ValueError('Flag --{} is required'.format(flag_name))

    annot_file = FLAGS.input_file
    output_file = FLAGS.output_file
    # Do not filter boxes using height
    all_filename, frame_to_object = read_annot(annot_file)
    print('Done loading annot.')
    if FLAGS.model_name == 'FasterRCNN':
        # new_frame_to_object = smooth(frame_to_object)
        print('smoothing FasterRCNN results')
        new_frame_to_object = smooth_annot(all_filename, frame_to_object)
        print('Done smoothing annot.')
        tag_object(all_filename, new_frame_to_object, output_file)
    else:
        # new_frame_to_object = smooth(frame_to_object)
        print('Done smoothing annot.')
        # tag_object(all_filename, new_frame_to_object, output_file, 0.8)
        tag_object(all_filename, frame_to_object, output_file, 0.8)
    print('Done smoothing and tagging annot.')


if __name__ == '__main__':
    app.run(main)
