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



def run_temporal(video_name, gt, dt, model, start_frame, end_frame, frame_rate, temporal_sampling_list, f_profile):
    f1_list = []
    for sample_rate in temporal_sampling_list:
        tpos = defaultdict(int)
        fpos = defaultdict(int)
        fneg = defaultdict(int)
        save_dt = []

        for img_index in range(start_frame, end_frame+1):

            # based on sample rate, decide whether this frame is sampled
            if img_index % sample_rate >= 1:
                # this frame is not sampled, so reuse the last saved
                # detection result
                continue
            else:
                # this frame is sampled, so use the full model result
                current_gt = gt[img_index]
                dt_boxes_final = dt[(model, img_index)]
                if current_gt == dt_boxes_final:
                    tpos[img_index] = 1
                else:
                    fpos[img_index] = 1


        tp_total = sum(tpos.values())
        fp_total = sum(fpos.values())
        fn_total = sum(fneg.values())

        f1_score = tp_total/(tp_total + fp_total)
        print('relative fps={}, f1={}'.format(1/sample_rate, f1_score))
        f1_list.append(f1_score)
        f_profile.write(','.join([video_name, str(1/sample_rate),
                                str(f1_score)])+'\n')
    return

def main():
    resol = '360p'
    dataset = 'cropped_crossroad4'
    short_video_length = 30
    metadata_file = '/mnt/data/zhujun/dataset/Youtube/' + dataset + '/metadata.json'
    dt_file = '/mnt/data/zhujun/dataset/Youtube/' + dataset + '/' + resol + '/profile/updated_gt_FasterRCNN_COCO_no_filter.csv'
    print("processing", dataset, dt_file)

    frame_rate = 30
    frame_count = 32000
    chunk_frame_cnt = short_video_length * frame_rate
    num_of_chunks = (frame_count-OFFSET*frame_rate)//chunk_frame_cnt
    model_list = ['mobilenet', 'inception', 'resnet50']
    dt = {}
    gt = {}
    with open(dataset + '_model_predictions.csv', 'r') as f:
        for line in f:
            line_list = line.strip().split(',')
            img_index = int(line_list[0])
            gt[img_index] = line_list[1]
            dt[('mobilenet', img_index)] = line_list[2]
            dt[('inception', img_index)] = line_list[3]
            dt[('resnet50', img_index)] = line_list[4]



    for model in model_list:
        f_out = open(dataset + '_' + model + '_model_wrt_temporal.csv', 'w')
        f_out.write("video_name,frame_rate,f1\n")

        for i in range(num_of_chunks):
            clip = dataset + '_' + str(i)
            # the 1st frame in the chunk
            start_frame = i * chunk_frame_cnt + 1 + OFFSET * frame_rate
            # the last frame in the chunk
            end_frame = (i + 1) * chunk_frame_cnt + OFFSET * frame_rate
            print('short video start={}, end={}'.format(start_frame,
                                                        end_frame))


            run_temporal(clip, gt, dt, model, start_frame, end_frame, 
                        frame_rate, TEMPORAL_SAMPLING_LIST, f_out)



if __name__ == '__main__':
    main()