"""VideoStorm Overfitting Script."""
import pdb
import argparse
import csv
import sys
import numpy as np
sys.path.append('/home/zhujunxiao/video_analytics_pipelines/code_repo/benchmarking/videostorm/')

from VideoStorm import VideoStorm
import copy
sys.path.append('/home/zhujunxiao/video_analytics_pipelines/code_repo/benchmarking/')
from video import YoutubeVideo
from collections import defaultdict
# from utils.model_utils import eval_single_image
from utils.utils import interpolation, compute_f1, IoU

from collections import defaultdict

TEMPORAL_SAMPLING_LIST = [1]
OFFSET = 0



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




def scale(box, in_resol, out_resol):
    """Scale the box at input resolution to output resolution.
    Args
        box: [x, y, w, h]
        in_resl: (width, height)
        out_resl: (width, height)
    """
    assert len(box) >= 4
    ret_box = box.copy()
    x_scale = out_resol[0]/in_resol[0]
    y_scale = out_resol[1]/in_resol[1]
    ret_box[0] = int(box[0] * x_scale)
    ret_box[1] = int(box[1] * y_scale)
    ret_box[2] = int(box[2] * x_scale)
    ret_box[3] = int(box[3] * y_scale)
    return ret_box


def scale_boxes(boxes, in_resol, out_resol):
    """Scale a list of boxes."""
    return [scale(box, in_resol, out_resol) for box in boxes]




def run_eval(video_name, gt, dt, model, resol, start_frame, end_frame, 
             original_video, video, f_profile):
    f1_list = []
    tpos = defaultdict(int)
    fpos = defaultdict(int)
    fneg = defaultdict(int)
    save_dt = []

    for img_index in range(start_frame, end_frame+1):
        # this frame is sampled, so use the full model result
        current_gt = copy.deepcopy(gt[img_index])
        dt_boxes_final = copy.deepcopy(dt[img_index])
        current_gt = scale_boxes(current_gt, original_video.resolution,
                                 video.resolution)        
        tpos[img_index], fpos[img_index], fneg[img_index] = \
                    eval_single_image(current_gt, dt_boxes_final)

        print(original_video.resolution, video.resolution)
    tp_total = sum(tpos.values())
    fp_total = sum(fpos.values())
    fn_total = sum(fneg.values())

    f1_score = compute_f1(tp_total, fp_total, fn_total)
    print('resol={}, model={}, f1={}'.format(video.resolution, model, f1_score))
    f_profile.write(','.join([video_name, resol, model,
                            str(f1_score)])+'\n')
    return

def main():
    dataset = 'driving_downtown_first'
    short_video_length = 30
    metadata_file = '/mnt/data/zhujun/dataset/Youtube/' + dataset + '/metadata.json'
    model_list = ['mobilenet', 'inception', 'Resnet50', 'FasterRCNN']
    resol_list = ['360p', '480p', '540p', '720p']
    f_out = open(dataset + 'spatial_wrt_model.csv', 'w')
    f_out.write("video_name, resol, model,f1\n")
          
    for model in model_list:
        original_resol = '720p'
        gt_file = '/mnt/data/zhujun/dataset/Youtube/' + dataset + '/' + original_resol + \
                    '/profile/updated_gt_' + model + '_COCO_no_filter.csv'
        original_video = YoutubeVideo(dataset, original_resol, metadata_file, gt_file,
                                    None, True)
        gt = original_video.get_video_detection() 
        frame_rate = original_video.frame_rate
        frame_count = original_video.frame_count
        chunk_frame_cnt = short_video_length * frame_rate
        num_of_chunks = (frame_count-OFFSET*frame_rate)//chunk_frame_cnt
        for resol in resol_list:
            dt_file = '/mnt/data/zhujun/dataset/Youtube/' + dataset + '/' + \
                        resol + '/profile/updated_gt_' + model + '_COCO_no_filter.csv'
            video = YoutubeVideo(dataset, resol, metadata_file, dt_file,
                                None, True)
            dt = video.get_video_detection() 
            for i in range(num_of_chunks):
                clip = dataset + '_' + str(i)
                # the 1st frame in the chunk
                start_frame = i * chunk_frame_cnt + 1 + OFFSET * frame_rate
                # the last frame in the chunk
                end_frame = (i + 1) * chunk_frame_cnt + OFFSET * frame_rate
                print('short video start={}, end={}'.format(start_frame,
                                                            end_frame))


                run_eval(clip, gt, dt, model, resol, start_frame, end_frame, 
                        original_video, video, f_out)



if __name__ == '__main__':
    main()