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

TEMPORAL_SAMPLING_LIST = [20, 15, 10, 5, 4, 3, 2.5, 2, 1.8, 1.5, 1.2, 1]

OFFSET = 0  # The time offset from the start of the video. Unit: seconds

image_resolution_dict = {'360p': [480, 360],
						 '480p': [640, 480],
						 '540p':[960, 540],
						 'walking': [3840,2160],
						 'driving_downtown': [3840, 2160], 
						 'highway': [1280,720],
						 'crossroad2': [1920,1080],
						 'crossroad': [1920,1080],
						 'crossroad3': [1280,720],
						 'crossroad4': [1920,1080],
						 'crossroad5': [1920,1080],
						 'driving1': [1920,1080],
						 'driving2': [1280,720],
                         'motorway': [1280,720],
						 'crossroad6':[1920,1080],
						 'crossroad7':[1920,1080],
						 'cropped_crossroad3':[600,400],
						 'cropped_driving2':[600,400]
						 }




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




def run_temporal(video_name, gt, dt, start_frame, end_frame, frame_rate, temporal_sampling_list, f_profile):
    f1_list = []
    for sample_rate in temporal_sampling_list:
        tpos = defaultdict(int)
        fpos = defaultdict(int)
        fneg = defaultdict(int)
        save_dt = []

        for img_index in range(start_frame, end_frame+1):
            dt_boxes_final = []
            current_full_model_dt = gt[img_index]
            current_gt = gt[img_index]
            # based on sample rate, decide whether this frame is sampled
            if img_index % sample_rate >= 1:
                # this frame is not sampled, so reuse the last saved
                # detection result
                continue
            else:
                # this frame is sampled, so use the full model result
                current_gt = gt[img_index].copy()
                dt_boxes_final = dt[img_index].copy()
                tpos[img_index], fpos[img_index], fneg[img_index] = \
                    eval_single_image(current_gt, dt_boxes_final)

        tp_total = sum(tpos.values())
        fp_total = sum(fpos.values())
        fn_total = sum(fneg.values())

        f1_score = compute_f1(tp_total, fp_total, fn_total)
        print('relative fps={}, f1={}'.format(1/sample_rate, f1_score))
        f1_list.append(f1_score)
        f_profile.write(','.join([video_name, str(1/sample_rate),
                                str(f1_score)])+'\n')
    return

def main():
    resol = '720p'
    dataset = 'motorway'
    short_video_length = 30
    image_resolution = image_resolution_dict[dataset]
    metadata_file = '/mnt/data/zhujun/dataset/Youtube/' + dataset + '/metadata.json'
    dt_file = '/mnt/data/zhujun/dataset/Youtube/' + dataset + '/' + resol + '/profile/updated_gt_FasterRCNN_COCO_no_filter.csv'
    print("processing", dataset, dt_file)
    video1 = YoutubeVideo(dataset, resol, metadata_file, dt_file,
                         None, True)
    frame_rate = video1.frame_rate
    frame_count = video1.frame_count
    chunk_frame_cnt = short_video_length * frame_rate
    num_of_chunks = (frame_count-OFFSET*frame_rate)//chunk_frame_cnt


    # gt = video1.get_video_detection()

    resol_list = ['540p', '480p', '360p']
    dt = {}

    for resol in resol_list:
        f_out = open(dataset + '_' + resol + '_spatial_wrt_temporal.csv', 'w')
        f_out.write("video_name,frame_rate,f1\n")
        dt_file = '/mnt/data/zhujun/dataset/Youtube/' + dataset + '/' + resol + '/profile/updated_gt_FasterRCNN_COCO.csv'
        gt_file = dt_file.replace('updated_gt_FasterRCNN_COCO.csv', 'gt_' + resol + '.csv')
        print("processing", dataset, dt_file)
        video = YoutubeVideo(dataset, resol, metadata_file, dt_file,
                            None, True)
        frame_rate = video.frame_rate
        frame_count = video.frame_count
        dt = video.get_video_detection()
        video1 = YoutubeVideo(dataset, resol, metadata_file, gt_file,
                            None, True)
        gt = video1.get_video_detection()




        
        for i in range(num_of_chunks):
            clip = dataset + '_' + str(i)
            # the 1st frame in the chunk
            start_frame = i * chunk_frame_cnt + 1 + OFFSET * frame_rate
            # the last frame in the chunk
            end_frame = (i + 1) * chunk_frame_cnt + OFFSET * frame_rate
            print('short video start={}, end={}'.format(start_frame,
                                                        end_frame))


            run_temporal(clip, gt, dt, start_frame, end_frame, 
                        frame_rate, TEMPORAL_SAMPLING_LIST, f_out)



if __name__ == '__main__':
    main()