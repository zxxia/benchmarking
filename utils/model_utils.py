from collections import defaultdict
import numpy as np
from utils.utils import IoU


def nonnegative(num):
    if num < 0:
        return 0.0
    return num


def load_jackson_detection(fullmodel_detection_path):
    full_model_dt = defaultdict(list)
    with open(fullmodel_detection_path, 'r') as f:
        f.readline()
        for line in f:
            img_idx, t, score, xmin, ymin, xmax, ymax = line.strip().split(',')
            # real image index starts from 1
            img_idx = int(img_idx)  # - 1
            xmin = nonnegative(float(xmin)) * 600
            ymin = nonnegative(float(ymin)) * 400
            xmax = nonnegative(float(xmax)) * 600
            ymax = nonnegative(float(ymax)) * 400
            # if not line_list[1]: # no detected object
            #     gt_boxes_final = []
            # else:
            #     gt_boxes_final = []
            #     for gt_box in gt_boxes:
            #         # t is object type
            #         tmp = [int(i) for i in gt_box.split(' ')]
            #         assert len(tmp) == 6, print(tmp, line)
            #         x = tmp[0]
            #         y = tmp[1]
            #         w = tmp[2]
            #         h = tmp[3]
            #         t = tmp[4]
            #         #if t == 3 or t == 8: # choose car and truch objects
            if (ymax-ymin) > 400/float(20) and (xmax - xmin) > 600/60:
                # ignore objects that are too small
                #         gt_boxes_final.append([x, y, x+w, y+h, t])
                full_model_dt[img_idx].append([xmin, ymin, xmax, ymax, t])

    return full_model_dt, img_idx


def load_ssd_detection(fullmodel_detection_path):
    full_model_dt = {}
    gt = {}
    img_list = []
    with open(fullmodel_detection_path, 'r') as f:
        for line in f:
            line_list = line.strip().split(',')
            img_index = int(line_list[0].split('.')[0])

            # if img_index > 5000: # test on ~3 mins
            #   break
            if not line_list[1]:  # no detected object
                dt_boxes_final = []
            else:
                dt_boxes_final = []
                dt_boxes = line_list[1].split(';')
                for dt_box in dt_boxes:
                    # t is object type
                    [x, y, w, h, t] = [int(i) for i in dt_box.split(' ')]
                    if t == 1:  # object is car, this depends on the task
                        dt_boxes_final.append([x, y, x+w, y+h, t])

            # load the ground truth
            if not line_list[2]:
                gt_boxes_final = []
            else:
                gt_boxes_final = []
                gt_boxes = line_list[2].split(';')
                for gt_box in gt_boxes:
                    # t is object type
                    [x, y, w, h, t] = [int(i) for i in gt_box.split(' ')]
                    if t == 1:  # object is car, this depends on the task
                        gt_boxes_final.append([x, y, x+w, y+h, t])

            img_list.append(img_index)
            full_model_dt[img_index] = dt_boxes_final
            gt[img_index] = gt_boxes_final

    return full_model_dt, gt, img_list


def load_kitti_ground_truth(filename, target_types=['Car']):
    '''
    Load kitti dataset ground truth (not the one detected by FasterRCNN)
    '''
    frame_to_obj = defaultdict(list)
    with open(filename, 'r') as f:
        # remove header
        f.readline()
        for line in f:
            cols = line.strip().split(',')
            frame_id = int(cols[0])
            obj_id = int(cols[1])
            x = int(cols[2])
            y = int(cols[3])
            w = int(cols[4])
            h = int(cols[5])
            t = str(cols[6])
            if t in target_types:
                frame_to_obj[frame_id].append([x, y, x+w, y+h, t, 1, obj_id])
    return frame_to_obj


def load_waymo_detection(fullmodel_detection_path):
    full_model_dt = {}
    with open(fullmodel_detection_path, 'r') as f:
        for line in f:
            line_list = line.strip().split(',')
            # real image index starts from 1
            img_index = int(line_list[0].split('.')[0])  # - 1
            if not line_list[1]:  # no detected object
                gt_boxes_final = []
            else:
                gt_boxes_final = []
                gt_boxes = line_list[1].split(';')
                for gt_box in gt_boxes:
                    # t is object type
                    box = gt_box.split(' ')
                    # if len(box) == 7, print(box, line)
                    w = int(float(box[2]))
                    h = int(float(box[3]))
                    x = int(float(box[0]) - float(box[2])/2)
                    y = int(float(box[1]) - float(box[3])/2)
                    t = int(box[4])
                    obj_id = box[-1]
                    if t == 1:  # choose car and truck objects
                        # if h > height/float(20):
                        # ignore objects that are too small
                        gt_boxes_final.append([x, y, x+w, y+h, 3, 1, obj_id])
            full_model_dt[img_index] = gt_boxes_final

    return full_model_dt, img_index


def load_full_model_detection(filename):
    '''
    Load full model detection results. This function should not filter any
    detections. The filter logic is separated to another function.
    '''
    full_model_dt = {}
    with open(filename, 'r') as f:
        for line in f:
            line_list = line.strip().split(',')
            # real image index starts from 1
            img_index = int(line_list[0].split('.')[0])  # - 1
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

    return full_model_dt, img_index


def filter_video_detections(video_detections, width_range=None,
                            height_range=None, target_types=None,
                            score_range=(0.0, 1.0)):
    """
    filter detection by height range, score range, types
    frame_detections: a dict mapping frame index to a list of object detections
                      each detection is in the format of list e.g.
                      [xmin, ymin, xmax, ymax, t, score, object id]
    height_range: a tuple (height_min, height_max) boudary included, it
                  restrains the height of bounding boxes. Default: None. No
                  height range limit.
    target_types: a set which contains the target types. Object with ypes not
                  in the set will be filered out. Default no types will be
                  filtered
    score_range: a tuple (score_min, score_max) boudary included, it
                  restrains the height of bounding boxes. Default: None. No
                  height range limit.

    return a dict which contains the filtered frame to object detections
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

    for frame_idx in video_detections:
        filtered_detections[frame_idx] = \
            filter_frame_detections(video_detections[frame_idx], width_range,
                                    height_range, target_types, score_range)

    return filtered_detections


def filter_frame_detections(detections, width_range=None, height_range=None,
                            target_types=None, score_range=(0.0, 1.0)):
    '''
    filter detection by height range, score range, types
    detections: a list of object detections each detection is in the format of
                list e.g.  [xmin, ymin, xmax, ymax, t, score, object id]
    height_range: a tuple (height_min, height_max) boudary included, it
                  restrains the height of bounding boxes. Default: None. No
                  height range limit.
    target_types: a set which contains the target types. Object with ypes not
                  in the set will be filered out. Default no types will be
                  filtered
    score_range: a tuple (score_min, score_max) boudary included, it
                  restrains the height of bounding boxes. Default: None. No
                  height range limit.

    return a dict which contains the filtered frame to object detections
    '''
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
    filtered_boxes = list()
    for box in detections:
        xmin, ymin, xmax, ymax, t, score, obj_id = box
        w = xmax - xmin
        h = ymax - ymin
        if target_types is not None and t not in target_types:
            continue
        if width_range is not None and \
           (w < width_range[0] or w > width_range[1]):
            continue
        if height_range is not None and \
           (h < height_range[0] or h > height_range[1]):
            continue
        if score_range is not None and \
           (score < score_range[0] or score > score_range[1]):
            continue
        filtered_boxes.append(box.copy())
    return filtered_boxes


def remove_overlappings(boxes, overlap_thr=None):
    """ to solve the occutation issue.
    remove the smaller box if two boxes overlap """
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
    """ compute the absolute area of a box in number of pixels """
    return (box[2]-box[0]+1) * (box[3]-box[1]+1)


def compute_intersection_area(box_i, box_j):
    """ compute the relative overlapping of the  """
    xmin = max(box_i[0], box_j[0])
    ymin = max(box_i[1], box_j[1])
    xmax = min(box_i[2], box_j[2])
    ymax = min(box_i[3], box_j[3])

    # compute the area of intersection rectangle
    inter_area = max(0, xmax-xmin+1) * max(0, ymax-ymin+1)

    return inter_area

def eval_single_image_single_type(gt_boxes, pred_boxes, iou_thresh):
    gt_idx_thr = []
    pred_idx_thr = []
    ious = []
    for ipb, pred_box in enumerate(pred_boxes):
        for igb, gt_box in enumerate(gt_boxes):
            iou = IoU(pred_box, gt_box)
            if iou > iou_thresh:
                gt_idx_thr.append(igb)
                pred_idx_thr.append(ipb)
                ious.append(iou)

    args_desc = np.argsort(ious)[::-1]
    if len(args_desc) == 0:
        # No matches
        tpos = 0
        fpos = len(pred_boxes)
        fneg = len(gt_boxes)
    else:
        gt_match_idx = []
        pred_match_idx = []
        for idx in args_desc:
            gt_idx = gt_idx_thr[idx]
            pr_idx = pred_idx_thr[idx]
            # If the boxes are unmatched, add them to matches
            if (gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
                gt_match_idx.append(gt_idx)
                pred_match_idx.append(pr_idx)
        tpos = len(gt_match_idx)
        fpos = len(pred_boxes) - len(pred_match_idx)
        fneg = len(gt_boxes) - len(gt_match_idx)
    return tpos, fpos, fneg


def eval_single_image(gt_boxes, dt_boxes, iou_thresh=0.5):
    tp_dict = {}
    fp_dict = {}
    fn_dict = {}
    gt = defaultdict(list)
    dt = defaultdict(list)
    for box in gt_boxes:
        gt[box[4]].append(box[0:4])
    for box in dt_boxes:
        dt[box[4]].append(box[0:4])

    for t in gt.keys():
        current_gt = gt[t]
        current_dt = dt[t]
        tp_dict[t], fp_dict[t], fn_dict[t] = \
            eval_single_image_single_type(current_gt, current_dt, iou_thresh)

    tp = sum(tp_dict.values())
    fp = sum(fp_dict.values())
    fn = sum(fn_dict.values())
    extra_t = [t for t in dt.keys() if t not in gt]
    for t in extra_t:
        # print('extra type', t)
        fp += len(dt[t])
    # print(tp, fp, fn)
    return tp, fp, fn
