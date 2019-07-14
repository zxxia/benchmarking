import cv2
import time
import numpy as np
from collections import defaultdict
import os
import json
from glimpse_kitti import pipeline, compute_target_frame_rate
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from my_utils import load_metadata
kitti = False
DATASET_LIST =  ['highway_no_traffic']#'jp','russia1', 'park']#['traffic', 'highway_normal_traffic', 'highway_no_traffic', 
                #'reckless_driving', 'motor', 'russia', 'jp_hw', 'tw_road', 
                #'tw_under_bridge']
PROFILE_LENGTH = 30 # 30 seconds
OFFSET = 1 # offset from the start of the video in unit of second
# standard_frame_rate = 10.0 # for comparison with KITTI
PATH = '/mnt/data/zhujun/new_video/'
SHORT_VIDEO_LENGTH = int(1*60)
INFERENCE_TIME = 100 # avg. GPU processing  time
TARGET_F1 = 0.9

# Two parameters: frame difference threshold, tracking error thresh
para1_list = [2,2.5,3,3.5,4,4.5,5,] # [2,3,4,5] #[1,2,5,10]#
para2_list = [15,11,10,8,5,4,3.5,3.1,3,1,0.6,0.5,0.3,0.1,0.05]
# [2.2,2.3,2.5,2.6] # [0.5,0.7,1,3,10]
#[2,3,5,10,] for russia1
#[0.5,0.55,0.6,0.7,]
# [10,12,14,15,20,25] for drift
# [1,1.5,1.8,2,3,4]
def get_gt_dt(annot_path, height):
    # read ground truth and full model (server side) detection results
    gt_annot = defaultdict(list)
    dt_annot = defaultdict(list)
    frame_end = 0
    with open(annot_path, 'r') as f:
        for line in f:
            annot_list = line.strip().split(',')
            frame_id = int(annot_list[0].replace('.jpg','')) 
            frame_end = frame_id
            gt_str = annot_list[1] # use full model detection results as ground truth
            gt_boxes = gt_str.split(';')
            if gt_boxes == ['']:
                gt_annot[frame_id] = []
            else:
                for box in gt_boxes:
                    box_list = box.split(' ')
                    x = int(box_list[0])
                    y = int(box_list[1])
                    w = int(box_list[2])
                    h = int(box_list[3])
                    t = int(box_list[4])
                    if t == 3 or t == 8: # object is car, this depends on the task
                        gt_annot[frame_id].append([x,y,w,h,t])
                        dt_annot[frame_id].append([x,y,w,h,t])

    return gt_annot, dt_annot, frame_end


def main():
    # final_result_f = open('youtube_glimpse_motivation.csv','w')
    # choose the first 3 mins to get the best frame diff thresh
    for video_type in DATASET_LIST:
        metadata = load_metadata(PATH + video_type + '/metadata.json')
        image_resolution = metadata['resolution'] 
        height = image_resolution[1]
        frame_rate = metadata['frame rate']
        frame_count = metadata['frame count']
        standard_frame_rate = frame_rate
       
        # resize the current video frames to 10Hz
        resize_rate = 1 
        # read ground truth and full model detection result
        # image name, detection result, ground truth
        annot_path = PATH + video_type + '/profile/updated_gt_FasterRCNN_COCO.csv'
        img_path = PATH + video_type +'/'
        gt_annot, dt_annot, frame_end = get_gt_dt(annot_path, height)

        num_of_short_videos = (frame_end - OFFSET * frame_rate)//(SHORT_VIDEO_LENGTH*frame_rate)
        
        test_f1_list = [] # A list of f1 scores computed using the best config over the entire video
        test_fps_list = [] # A list of fps computed over the entire video using the best config

        with open('glimpse_motivation_result_{}.csv'.format(video_type), 'w') as final_result_f:
            final_result_f.write('video chunk,para1,para2,f1,frame rate\n')
            for i in range(num_of_short_videos):
                start = i * (SHORT_VIDEO_LENGTH*frame_rate) + 1 + OFFSET * frame_rate

                end = (i + 1) * (SHORT_VIDEO_LENGTH*frame_rate) + OFFSET * frame_rate
                profile_start = start
                profile_end = start + PROFILE_LENGTH*frame_rate
                print("short video start={}, end={}".format(start, end))
                print("profile start={}, end={}".format(profile_start, profile_end))
                print('profiling short video {}'.format(i))
                # print(video_type, seg_index)
                # Run inference on the first 30s video
                frame_rate_list = []
                f1_list = []
                min_f1_gt_target = 1.0 # the minimum f1 score which is greater than
                                       # or equal to target f1(e.g. 0.9)
                min_fps = frame_rate

                best_para1 = -1
                best_para2 = -1
                is_tuning = True
                for para1 in para1_list:
                    if not is_tuning:
                        break
                    for para2 in para2_list:
                        if not is_tuning:
                            break
                        csvf = open('no_meaning.csv','w')
                        # larger para1, smaller thresh, easier to be triggered
                        frame_difference_thresh = image_resolution[0]*image_resolution[1]/para1
                        tracking_error_thresh = para2
                        # images start from index 1
                        print(img_path, profile_start, profile_end) 
                        triggered_frame, f1 = pipeline(img_path, dt_annot, gt_annot,
                                                       profile_start, profile_end, csvf, 
                                                       image_resolution, frame_rate, 
                                                       frame_difference_thresh, 
                                                       tracking_error_thresh, False)

                        current_frame_rate = triggered_frame / float(PROFILE_LENGTH)
                        frame_rate_list.append(current_frame_rate)
                        f1_list.append(f1)

                        if f1 >= TARGET_F1 and f1 <=min_f1_gt_target  and current_frame_rate < min_fps:
                            # record min f1 which is greater than target f1
                            min_f1_gt_target = f1
                            min_fps = current_frame_rate
                            # record the best config
                            best_para1 = para1
                            best_para2 = para2
                            is_tuning = False 

                        print('para1 = {}, para2 = {}, Profiled f1 = {}, Profiled frame rate = {}'.format(para1, para2, f1, current_frame_rate))

                print("Finish profiling on the {} {}s segment...".format(i, PROFILE_LENGTH))
                print("best_para1 = {}, best_para2={}".format(best_para1, best_para2))
                print("Start testing....")
                # use the selected parameters for the next 5 mins
                frame_difference_thresh = image_resolution[0]*image_resolution[1]/best_para1   
                tracking_error_thresh = best_para2

                assert(end>=1 and end <= frame_count)
                csvf = open('no_meaning.csv','w')
                triggered_frame, f1 = pipeline(img_path, dt_annot, gt_annot, start,
                                               end, csvf, image_resolution,
                                               frame_rate, frame_difference_thresh, 
                                               tracking_error_thresh, False)
                print("triggered {} frames over {} seconds".format(triggered_frame, SHORT_VIDEO_LENGTH))
                fps = float(triggered_frame)/float(SHORT_VIDEO_LENGTH)
                test_f1_list.append(f1)
                test_fps_list.append(fps)
                print('tested F1, tested frame_rate:', f1, fps)
                final_result_f.write(video_type + '_' + str(i) + ',' + 
                      str(best_para1) + ',' + str(best_para2) + ',' +
                      str(f1) + ',' + str(fps/frame_rate) + '\n')
                print("Finished testing...")

        test_relative_fps_list = [x/frame_rate for x in test_fps_list]
        if test_relative_fps_list and test_f1_list:
            plt.scatter(test_relative_fps_list, test_f1_list, label=video_type)
    plt.xlabel("GPU Processing time")
    plt.ylabel("Accuracy")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("Glimpse Motivation")
    plt.legend()
    plt.savefig("/home/zxxia/figs/glimpse/glimpse_motivation.png")
    #plt.clf()
    #plt.show()


if __name__=='__main__':
    main()
