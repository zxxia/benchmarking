"""Functions to evaluate the performance of a pipeline."""
from collections import defaultdict

import numpy as np


def IoU(boxA, boxB):
    """Return IoU of two boxes."""
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def evaluate_single_type(gt_boxes, pred_boxes, iou_thresh):
    """Compute the tp, fp, and fn of object detection on a frame.

    Args
        gt_boxes(list): groundtruth, a python list of bounding boxes.
        dt_boxes(list): detections, a python list of bounding boxes.
        iou_thresh(float): iou threshold.

    Return
        tp(int): True positive.
        fp(int): False positive.
        fn(int): False negative.

    """
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


def evaluate_frame(gt_boxes, dt_boxes, iou_thresh=0.5):
    """Compute the tp, fp, and fn of object detection on a frame.

    Args
        gt_boxes(list): groundtruth, a python list of bounding boxes.
        dt_boxes(list): detections, a python list of bounding boxes.
        iou_thresh(float): iou threshold.

    Return
        tp(int): True positive.
        fp(int): False positive.
        fn(int): False negative.

    """
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
        tp_dict[t], fp_dict[t], fn_dict[t] = evaluate_single_type(
            gt[t], dt[t], iou_thresh)

    tp = sum(tp_dict.values())
    fp = sum(fp_dict.values())
    fn = sum(fn_dict.values())
    # extra types detected
    extra_t = [t for t in dt.keys() if t not in gt]
    for t in extra_t:
        fp += len(dt[t])
    return tp, fp, fn


def evaluate_video(gt_boxes, dt_boxes, iou_thresh=0.5):
    """Compute the f1 score of object detection on a video.

    Args
        gt_boxes(dict): groundtruth, a python dictionary mapping frame id to a
                        list of bounding boxes.
        dt_boxes(dict): detections, a python dictionary mapping frame id to a
                        list of bounding boxes.
        iou_thresh(float): iou threshold.

    Return
        tp(int): True positive.
        fp(int): False positive.
        fn(int): False negative.

    """
    tp_tot = 0
    fp_tot = 0
    fn_tot = 0
    for frame_id in sorted(gt_boxes):
        # assert frame_id in gt_boxes
        # assert frame_id in dt_boxes, f'{frame_id} not in det'
        if frame_id not in dt_boxes:
            dt_boxes[frame_id] = []
        tp, fp, fn = evaluate_frame(
            gt_boxes[frame_id], dt_boxes[frame_id], iou_thresh)
        tp_tot += tp
        fp_tot += fp
        fn_tot += fn

    return tp_tot, fp_tot, fn_tot


def compute_f1(tp, fp, fn):
    """Compute F1 score.

    Args
        tp(int): True positive.
        fp(int): False positive.
        fn(int): False negative.

    Return
        f1(float): F1 score.

    """
    if tp:
        precison = float(tp) / (tp + fp)
        recall = float(tp) / (tp + fn)
        f1 = 2*(precison*recall)/(precison+recall)
    else:
        if fn:
            f1 = 0
        else:
            f1 = 1
    return f1
