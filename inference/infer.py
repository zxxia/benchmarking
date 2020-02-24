"""Object detection module."""
import csv
import glob
import os
import pdb
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

from benchmarking.utils.utils import nms


def load_model(model_dir):
    """Load model downloaded from tensorflow model zoo."""
    model_dir = os.path.join(model_dir, "saved_model")
    model = tf.saved_model.load(str(model_dir))
    model = model.signatures['serving_default']
    return model


def run_inference_for_single_image(model, image):
    """Run object detection on a single image."""
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    output_dict = model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and
    # take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(
        np.int64)

    return output_dict


def write_to_file(csvwriter, img_path, resolution, detections):
    """Write detetions into a csv file."""
    for label, box, score in zip(detections['detection_classes'],
                                 detections['detection_boxes'],
                                 detections['detection_scores']):
        csvwriter.writerow(
            [int(os.path.basename(os.path.splitext(img_path)[0])),
             box[1]*resolution[0], box[0]*resolution[1],
             box[3]*resolution[0], box[2]*resolution[1], label, score])


def load_object_detection_results(filename):
    """Load object detection results.

    filename(string): filename of object detection results in csv format.
    return(dict): a dict mapping frame id to ['xmin', 'ymin', 'xmax', 'ymax',
                                              'class', 'score', 'object id']
    """
    df = pd.read_csv(filename)
    if 'object id' in df.columns:
        dets = {k: g[['xmin', 'ymin', 'xmax', 'ymax', 'class',
                      'score', 'object id']
                     ].values.tolist() for k, g in df.groupby("frame id")}
    else:
        dets = {k: g[['xmin', 'ymin', 'xmax', 'ymax', 'class', 'score']
                     ].values.tolist() for k, g in df.groupby("frame id")}
    return dets


def tag_object(dets, dist_coef=0.45):
    """Assign object id to bounding boxes.

    Args
        frame_to_object: a dict in which frame id maps to object detections
        output_file and update_gt_file stores the detections with object id
        but in different format
        dist_coef: adjust the threshold of distance which is used to determine
        whether two bounding boxes belong to the same object, if use FasterRCNN
        dist_coef should be set to 0.45. Otherwise, mobilenet needs to be 0.8
    Return

    """
    last_frame = []
    object_to_type = defaultdict(list)
    object_cn = 0
    object_id = 0
    new_dets = {}
    #     f.write('Frame ID, Object ID, X, Y, W, H, Object Type, Score\n')
    # with open(update_gt_file, 'w') as gt_f:
    for frame_id in sorted(dets):
        boxes = dets[frame_id]
        new_dets[frame_id] = []
        current_frame = []
        for box in boxes:
            try:
                assert len(box) == 6, print(frame_id, boxes)
            except AssertionError:
                pdb.set_trace()
            # find the center point of the bounding box
            x_c = (box[0] + box[2])/2.0
            y_c = (box[1] + box[3])/2.0
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
                else:
                    # old object, get the id
                    object_id = last_frame[index][2]
                    last_frame.remove(last_frame[index])
            object_to_type[object_id].append(box[4])
            new_dets[frame_id].append(box+[object_id])
            current_frame.append([x_c, y_c, object_id])

        last_frame = current_frame
    # update the type for each object
    for object_id in object_to_type.keys():
        types = object_to_type[object_id]
        type_counter = Counter(types)
        object_to_type[object_id] = type_counter.most_common(1)[0][0]

    for frame_id in sorted(new_dets):
        boxes = new_dets[frame_id]
        for box_idx, box in enumerate(boxes):
            assert len(box) == 7, print(box)
            object_id = box[-1]
            box[4] = object_to_type[object_id]

            new_dets[frame_id][box_idx] = box
    return new_dets


def smooth_annot(dets, dist_coef=0.45):
    """Smooth object detection results.

    For each frame, if a box exists in last frame and next frame
    but not in current frame, add it to current frame
    update_gt_file = output_folder + 'updated_gt_FasterRCNN_COCO.csv'.
    """
    for frame_id in dets:
        boxes = dets[frame_id]
        keep = nms(np.array(boxes), 0.4)
        boxes = [boxes[i] for i in keep]
        dets[frame_id] = boxes

    for frame_id in sorted(dets):
        if frame_id - 1 not in dets or not dets[frame_id-1] or \
                frame_id+1 not in dets or not dets[frame_id+1]:
            current_boxes = []
        else:
            current_boxes = dets[frame_id]
            last_boxes = dets[frame_id-1] if frame_id - 1 in dets else []
            next_boxes = dets[frame_id+1] if frame_id + 1 in dets else []
            try:
                assert len(last_boxes) and len(next_boxes), print(frame_id)
            except AssertionError:
                pdb.set_trace()
            for box in next_boxes:
                x_c = (box[0] + box[2])/2.0
                y_c = (box[1] + box[3])/2.0
                thresh = dist_coef * box[3]
                dist_with_last_frame = []
                dist_with_current_frame = []
                # check if this box exists in last frame
                for last_box in last_boxes:
                    last_x_c = (last_box[0] + last_box[2])/2.0
                    last_y_c = (last_box[1] + last_box[3])/2.0
                    dist = np.linalg.norm(np.array([x_c, y_c]) -
                                          np.array([last_x_c, last_y_c]))
                    dist_with_last_frame.append(dist)
                min_last_dist = min(dist_with_last_frame)

                # check if this box exists in current frame
                if not current_boxes:
                    min_current_dist = 100
                else:
                    for current_box in current_boxes:
                        current_x_c = (current_box[0] + current_box[2])/2.0
                        current_y_c = (current_box[1] + current_box[3])/2.0
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
                x_c = (box[0] + box[2])/2.0
                y_c = (box[1] + box[3])/2.0
                thresh = dist_coef * box[3]
                dist_with_last_frame = []
                dist_with_next_frame = []
                # check if this box exists in last frame
                for last_box in last_boxes:
                    last_x_c = (last_box[0] + last_box[2])/2.0
                    last_y_c = (last_box[1] + last_box[3])/2.0
                    dist = np.linalg.norm(
                        np.array([x_c, y_c]) - np.array([last_x_c, last_y_c]))
                    dist_with_last_frame.append(dist)
                min_last_dist = min(dist_with_last_frame)

                # check if this box exists in current frame

                for next_box in next_boxes:
                    next_x_c = (next_box[0] + next_box[2])/2.0
                    next_y_c = (next_box[1] + next_box[3])/2.0
                    dist = np.linalg.norm(
                        np.array([x_c, y_c]) - np.array([next_x_c, next_y_c]))
                    dist_with_next_frame.append(dist)

                min_next_dist = min(dist_with_next_frame)

                # if this box exist in last frame, but not in current frame
                # add this box to current frame
                if min_last_dist > thresh and min_next_dist > thresh:
                    assert len(box) == 6, print(box)
                    current_boxes.remove(box)

                dets[frame_id] = current_boxes

    return dets


def infer(args):
    """Do object detection."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    assert len(gpus) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_visible_devices(gpus[args.device], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[args.device], True)
    model = load_model(args.model)

    dets_path = os.path.join(args.output_path, 'detections.csv')
    with open(dets_path, 'w', 1) as f:
        writer = csv.writer(f)
        writer.writerow(['frame id', 'xmin', 'ymin',
                         'xmax', 'ymax', 'class', 'score'])
        img_paths = sorted(glob.glob(os.path.join(args.input_path, '*.jpg')))
        for i, img_path in enumerate(img_paths):
            image = np.array(Image.open(img_path))
            detections = run_inference_for_single_image(model, image)
            write_to_file(writer, img_path, image.shape, detections)
            if (i+1) % 10 == 0 or (i+1) == len(img_paths):
                print('Processed {}/{} images...'.format(i+1, len(img_paths)))

        img_paths = sorted(glob.glob(os.path.join(args.input_path, '*.png')))
        for i, img_path in enumerate(img_paths):
            image = np.array(Image.open(img_path))
            detections = run_inference_for_single_image(model, image)
            write_to_file(writer, img_path, image.shape, detections)
            if (i+1) % 10 == 0 or (i+1) == len(img_paths):
                print('Processed {}/{} images...'.format(i+1, len(img_paths)))

    dets = load_object_detection_results(dets_path)
    dets = smooth_annot(dets)
    dets = tag_object(dets)
    smoothed_dets_path = os.path.join(
        args.output_path, 'smoothed_detections.csv')
    with open(smoothed_dets_path, 'w', 1) as f:
        writer = csv.writer(f)
        writer.writerow(['frame id', 'xmin', 'ymin',
                         'xmax', 'ymax', 'class', 'score', 'object id'])
        for frame_id in sorted(dets):
            for box in dets[frame_id]:
                writer.writerow([frame_id]+box)