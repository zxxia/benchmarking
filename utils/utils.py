"""Commonly Used functions."""
import json
from collections import Counter


def load_metadata(filename):
    with open(filename) as f:
        metadata = json.load(f)
    return metadata


def interpolation(point_a, point_b, target_x):
    k = float(point_b[1] - point_a[1])/(point_b[0] - point_a[0])
    b = point_a[1] - point_a[0]*k
    target_y = k*target_x + b
    return target_y


def Most_Common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]


def load_full_model_detection(filename):
    """Load full model detection results.

    This function should not filter any detections. The filter logic is
    separated to another function.
    """
    full_model_dt = {}
    with open(filename, 'r') as f:
        for line in f:
            line_list = line.strip().split(',')
            # real image index starts from 1
            img_index = int(line_list[0].split('.')[0])
            if not line_list[1]:  # no detected object
                gt_boxes_final = []
            else:
                gt_boxes_final = []
                gt_boxes = line_list[1].split(';')
                for gt_box in gt_boxes:
                    # t is object type
                    box = gt_box.split(' ')
                    assert len(box) == 7,  \
                        "the length of the detection is not 7." \
                        " some data is missing"
                    x = int(box[0])
                    y = int(box[1])
                    w = int(box[2])
                    h = int(box[3])
                    t = int(box[4])
                    score = float(box[5])
                    obj_id = int(box[6])
                    gt_boxes_final.append([x, y, x+w, y+h, t, score, obj_id])
            full_model_dt[img_index] = gt_boxes_final

    return full_model_dt


def filter_video_detections(video_detections, width_range=None,
                            height_range=None, target_types=None,
                            score_range=(0.0, 1.0)):
    """Filter video detection by height range, score range, types.

    Args
        frame_detections: a dict mapping frame index to a list of object
                          detections each detection is in the format of list
                          e.g.  [xmin, ymin, xmax, ymax, t, score, object id]
        height_range: a tuple (height_min, height_max) boudary included, it
                      restrains the height of bounding boxes. Default: None. No
                      height range limit.
        target_types: a set which contains the target types. Object with ypes
                      not in the set will be filered out. Default no types will
                      be filtered
        score_range: a tuple (score_min, score_max) boudary included, it
                      restrains the height of bounding boxes. Default: None. No
                      height range limit.

    Return
        a dict which contains the filtered frame to object detections
        a dict which contains the discarded frame to object detections

    """
    assert (width_range is None) or \
           (isinstance(width_range, tuple) and len(width_range) == 2 and
            width_range[0] <= width_range[1]), \
        'width range needs to be a length 2 tuple or None'
    assert (height_range is None) or \
           (isinstance(height_range, tuple) and len(height_range) == 2 and
            height_range[0] <= height_range[1]), \
        'height range needs to be a length 2 tuple or None'
    assert (target_types is None) or isinstance(target_types, set), \
        "target_types need to be a set or None"
    assert (isinstance(score_range, tuple) and len(score_range) == 2 and
            score_range[0] <= score_range[1]), \
        'score range needs to be a length 2 tuple or None'
    filtered_detections = dict()
    discarded_detections = dict()

    for frame_idx in video_detections:
        filtered_detections[frame_idx], discarded_detections[frame_idx] = \
            filter_frame_detections(video_detections[frame_idx], width_range,
                                    height_range, target_types, score_range)

    return filtered_detections, discarded_detections


def filter_frame_detections(detections, width_range=None, height_range=None,
                            target_types=None, score_range=(0.0, 1.0)):
    """Filter frame detection by height range, score range, types.

    Args
        detections: a list of object detections each detection is in the format
                    of list e.g.  [xmin, ymin, xmax, ymax, t, score, object id]
        height_range: a tuple (height_min, height_max) boudary included, it
                      restrains the height of bounding boxes. Default: None. No
                      height range limit.
        target_types: a set which contains the target types. Object with ypes
                      not in the set will be filered out. Default no types will
                      be filtered
        score_range: a tuple (score_min, score_max) boudary included, it
                      restrains the height of bounding boxes. Default: None. No
                      height range limit.

    Return
        a list of object detection boxes that pass filter
        a list of object detection boxes that do not pass filter

    """
    assert (width_range is None) or \
           (isinstance(width_range, tuple) and len(width_range) == 2 and
            width_range[0] <= width_range[1]), \
        'width range needs to be a length 2 tuple or None'
    assert (height_range is None) or \
           (isinstance(height_range, tuple) and len(height_range) == 2 and
            height_range[0] <= height_range[1]), \
        'height range needs to be a length 2 tuple or None'
    assert (target_types is None) or isinstance(target_types, set), \
        "target_types need to be a set or None"
    assert (isinstance(score_range, tuple) and len(score_range) == 2 and
            score_range[0] <= score_range[1]), \
        'score range needs to be a length 2 tuple or None'
    kept_boxes = list()
    discarded_boxes = list()
    for box in detections:
        xmin, ymin, xmax, ymax, t, score, obj_id = box
        w = xmax - xmin
        h = ymax - ymin
        if target_types is not None and t not in target_types:
            discarded_boxes.append(box.copy())
            continue
        if width_range is not None and \
           (w < width_range[0] or w > width_range[1]):
            discarded_boxes.append(box.copy())
            continue
        if height_range is not None and \
           (h < height_range[0] or h > height_range[1]):
            discarded_boxes.append(box.copy())
            continue
        if score_range is not None and \
           (score < score_range[0] or score > score_range[1]):
            discarded_boxes.append(box.copy())
            continue
        kept_boxes.append(box.copy())
    return kept_boxes, discarded_boxes


def remove_overlappings(boxes, overlap_thr=None):
    """Attempt to solve the occutation issue.

    Remove the smaller box if two boxes overlap.
    """
    # sort all boxes based on area
    if overlap_thr is None:
        return boxes
    assert 0 <= overlap_thr <= 1.0
    sorted_boxes = sorted(boxes, key=compute_area, reverse=True)
    idx_2_remove = set()
    for i, box_i in enumerate(sorted_boxes):
        # if i in indices_2_remove:
        #     continue
        for j in range(i+1, len(sorted_boxes)):
            area_j = compute_area(sorted_boxes[j])
            inter = compute_intersection_area(box_i, sorted_boxes[j])
            if inter/area_j > overlap_thr:  # for sure area i >= area_j
                idx_2_remove.add(j)
    ret = [box for i, box in enumerate(sorted_boxes) if i not in idx_2_remove]
    return ret


def compute_area(box):
    """Compute the absolute area of a box in number of pixels."""
    return (box[2]-box[0]+1) * (box[3]-box[1]+1)


def compute_intersection_area(box_i, box_j):
    """Compute the relative overlapping of two boxes."""
    xmin = max(box_i[0], box_j[0])
    ymin = max(box_i[1], box_j[1])
    xmax = min(box_i[2], box_j[2])
    ymax = min(box_i[3], box_j[3])

    # compute the area of intersection rectangle
    inter_area = max(0, xmax-xmin+1) * max(0, ymax-ymin+1)

    return inter_area


def convert_detection_to_classification(video_detections,
                                        resolution,
                                        size_thresh=0.005,
                                        confidence_thresh=0.85):
    """Convert detection boxes to per-frame classification label.
        (1) Find the largest box.
        (2) Use the label for that box as the label for this frame.
    """
    cocomap_file = '../ground_truth/mscoco_label_map.pbtxt'
    COCO_map = load_COCOlabelmap(cocomap_file)
    video_classification_label = {}
    for frame_idx in video_detections:
        frame_label = convert_frame_label(
            video_detections[frame_idx], COCO_map, resolution, size_thresh,
            confidence_thresh)
        video_classification_label[frame_idx] = frame_label

    return video_classification_label


def convert_frame_label(boxes, COCO_map, resolution, size_thresh,
                        confidence_thresh):
    """Return the label of largest box.
    Arguments:
        boxes {[dict]} -- detection boxes for current frame.
        Each box is a list, [x1, y1, x2, y2, t, score, obj_id]
    """
    area = []
    label = []
    for box in boxes:

        confidence = box[5]
        size = (box[2] - box[0]) * (box[3] - box[1]) / \
            (resolution[0] * resolution[1])
        if confidence < confidence_thresh or size < size_thresh:
            continue
        else:
            assert box[2] > box[0], print('wrong box', box)
            area.append(size)
            label.append(COCO_map[box[4]])

    if area == []:
        final_label = ['no_object', 1.0]
    else:
        index = area.index(max(area))
        final_label = [label[index], max(area)]
    return final_label


def smooth_classification(labels):
    smoothed_labels = {}
    # current we are using traffic videos
    all_classes = ['car', 'person', 'truck',
                   'bicycle', 'bus', 'motorcycle', 'no_object']
    # if those labels appear, map them to 'car'
    label_to_car = ['bed']
    # if those labels appear, map them to 'truck'
    label_to_truck = ['airplane', 'boat', 'train',
                      'suitcase', 'bus', 'bench', 'book']
    for frame_idx in labels:
        if labels[frame_idx][0] in label_to_car:
            smoothed_labels[frame_idx] = ['car', labels[frame_idx][1]]
        elif labels[frame_idx][0] in label_to_truck:
            smoothed_labels[frame_idx] = ['truck', labels[frame_idx][1]]
        elif labels[frame_idx][0] not in all_classes:
            smoothed_labels[frame_idx] = ['no_object', 1.0]
        else:
            smoothed_labels[frame_idx] = labels[frame_idx]

    for frame_idx in range(min(smoothed_labels) + 1, max(smoothed_labels)):
        if smoothed_labels[frame_idx - 1][0] == smoothed_labels[frame_idx + 1][0]:
            if smoothed_labels[frame_idx][0] != smoothed_labels[frame_idx - 1][0]:
                smoothed_labels[frame_idx] = smoothed_labels[frame_idx - 1]

    return smoothed_labels


def load_COCOlabelmap(label_map_path):
    COCO_Labelmap = {}
    COCO_Labelmap_reverse = {}
    with open(label_map_path, 'r') as f:
        line = f.readline()
        while line:
            if 'id' in line:
                ID = int(line.strip().split(':')[1].strip())
                line = f.readline()
                label = line.strip().split(':')[1]
                COCO_Labelmap[ID] = label.strip().replace('"', '')
                COCO_Labelmap_reverse[label.strip().replace('"', '')] = ID
                line = f.readline()
            else:
                line = f.readline()

    return COCO_Labelmap, COCO_Labelmap_reverse


def read_json_file(filename):
    """Load json object from a file."""
    with open(filename, 'r') as f:
        content = json.load(f)
    return content


def write_json_file(filename, content):
    """Dump into a json file."""
    with open(filename, 'w') as f:
        json.dump(content, f, indent=4)
