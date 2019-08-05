from collections import defaultdict
import numpy as np
from utils.utils import IoU

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
            if not line_list[1]: # no detected object
                dt_boxes_final = []
            else:
                dt_boxes_final = []
                dt_boxes = line_list[1].split(';')
                for dt_box in dt_boxes:
                    # t is object type
                    [x, y, w, h, t] = [int(i) for i in dt_box.split(' ')]
                    if t == 1: # object is car, this depends on the task
                        dt_boxes_final.append([x, y, x+w, y+h])

            # load the ground truth
            if not line_list[2]:
                gt_boxes_final = []
            else:
                gt_boxes_final = []
                gt_boxes = line_list[2].split(';')
                for gt_box in gt_boxes:
                    # t is object type
                    [x, y, w, h, t] = [int(i) for i in gt_box.split(' ')]
                    if t == 1: # object is car, this depends on the task
                        gt_boxes_final.append([x, y, x+w, y+h])                 
            
            img_list.append(img_index)
            full_model_dt[img_index] = dt_boxes_final
            gt[img_index] = gt_boxes_final
   
    return full_model_dt, gt, img_list


def load_full_model_detection(fullmodel_detection_path):#, height):
    full_model_dt = {}
    with open(fullmodel_detection_path, 'r') as f:
        for line in f:
            line_list = line.strip().split(',')
            # real image index starts from 1
            img_index = int(line_list[0].split('.')[0]) #- 1
            if not line_list[1]: # no detected object
                gt_boxes_final = []
            else:
                gt_boxes_final = []
                gt_boxes = line_list[1].split(';')
                for gt_box in gt_boxes:
                    # t is object type
                    tmp = [int(i) for i in gt_box.split(' ')]
                    assert len(tmp) == 6, print(tmp, line)
                    x = tmp[0]
                    y = tmp[1]
                    w = tmp[2]
                    h = tmp[3]
                    t = tmp[4]
                    if t == 3 or t == 8: # choose car and truch objects
                        # if h > height/float(20): # ignore objects that are too small
                        gt_boxes_final.append([x, y, x+w, y+h, 3])
            full_model_dt[img_index] = gt_boxes_final
            
    return full_model_dt, img_index

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
        tp = 0
        fp = len(pred_boxes)
        fn = len(gt_boxes)
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
        tp = len(gt_match_idx)
        fp = len(pred_boxes) - len(pred_match_idx)
        fn = len(gt_boxes) - len(gt_match_idx)
    return tp, fp, fn

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
        tp_dict[t], fp_dict[t], fn_dict[t] = eval_single_image_single_type(
                                             current_gt, current_dt, iou_thresh)

    tp = sum(tp_dict.values())
    fp = sum(fp_dict.values())
    fn = sum(fn_dict.values())
    extra_t = [t for t in dt.keys() if t not in gt]
    for t in extra_t:
        print('extra type', t)
        fp += len(dt[t])
    #print(tp, fp, fn)
    return tp, fp, fn

