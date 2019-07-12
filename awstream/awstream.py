from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import sys
import json
from videostorm.VideoStorm_temporal import load_full_model_detection, eval_single_image
from my_utils import interpolation, compute_f1
import cv2
import os

PATH = '/home/zxxia/videos/'
TEMPORAL_SAMPLING_LIST = [20,15,10,5,4,3,2.5,2,1.8,1.5,1.2,1]
DATASET_LIST = ['traffic','highway_normal_traffic', 'highway_no_traffic',
                'reckless_driving', 'motor', 'jp_hw', 'russia', 'tw_road',
                'tw_under_bridge']
IMAGE_RESOLUTION_DICT = {'360p': [480, 360],
                         '480p': [640, 480],
                         '540p':[960, 540],
                        }

SHORT_VIDEO_LENGTH = 2*60 # divide each video into 5-min
IOU_THRESH = 0.5
TARGET_F1 = 0.9


def compute_video_size(dataset, start, end, 
    target_frame_rate, frame_rate, standard_frame_rate, resolution):
    with open(PATH + dataset + '/metadata.json') as metadata_f:
        metadata = json.load(metadata_f)

    img_path = PATH + dataset
    frame_array = []
    sample_rate = standard_frame_rate/target_frame_rate
    img_cn = 0

    for img_index in range(start, end):
         
        resize_rate = frame_rate/standard_frame_rate
        if img_index%resize_rate >= 1:
            continue
        else:
            img_index = img_cn
            img_cn += 1

        # based on sample rate, decide whether this frame is sampled
        if img_index%sample_rate >= 1:
            continue
        else:
            if resolution == 'original':
                filename = img_path + '/' + format(img_index+1, '06d') + '.jpg'
                image_resolution = metadata['resolution']
            else:
                filename = img_path + '/' + resolution + '/' + \
                format(img_index+1, '06d') + '.jpg'
                image_resolution = IMAGE_RESOLUTION_DICT[resolution]
        img = cv2.imread(filename)
        height, width, layers = img.shape
        assert height == image_resolution[1] and width == image_resolution[0], print(filename, height, width)

        frame_array.append(img)
    print(target_frame_rate, image_resolution, len(frame_array))
    out = cv2.VideoWriter('tmp.mp4', cv2.VideoWriter_fourcc(*'MJPG'), int(target_frame_rate), (image_resolution[0], image_resolution[1]))

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()
    video_size = os.path.getsize("tmp.mp4")
    print(target_frame_rate, image_resolution, video_size)
    return video_size


def profile(dataset, frame_rate, gt, start_frame, chunk_length=30):
    result = {}
    
    with open(PATH + dataset + '/metadata.json') as metadata_f:
        metadata = json.load(metadata_f)

    standard_frame_rate = frame_rate
    # choose resolution
    resolution_list = ['original','540p', '480p', '360p']
    # choose frame rate
    for resolution in resolution_list:
        F1_score_list = []
        if resolution == 'original':
            dt_file = PATH + dataset + '/profile/updated_gt_FasterRCNN_COCO.csv' 
            image_resolution = metadata['resolution']
        else:
            dt_file = PATH + dataset + '/' + resolution + '/profile/updated_gt_FasterRCNN_COCO.csv'
            image_resolution = IMAGE_RESOLUTION_DICT[resolution]
            gt_file = dt_file.replace('updated_gt_FasterRCNN_COCO.csv', 'gt_' + resolution + '.csv')
            gt, _ = load_full_model_detection(gt_file, height)      


        height = image_resolution[1]
        full_model_dt, num_of_frames = load_full_model_detection(dt_file, height)               
        for sample_rate in TEMPORAL_SAMPLING_LIST:
            img_cn = 0
            tp = defaultdict(int)
            fp = defaultdict(int)
            fn = defaultdict(int)
            save_dt = []
            for img_index in range(start_frame, start_frame+chunk_length*frame_rate):
                dt_boxes_final = []
                current_full_model_dt = full_model_dt[img_index]
                current_gt = gt[img_index]
                resize_rate = frame_rate/standard_frame_rate
                if img_index%resize_rate >= 1:
                    continue
                else:
                    img_index = img_cn
                    img_cn += 1

                # based on sample rate, decide whether this frame is sampled
                if img_index%sample_rate >= 1:
                    # this frame is not sampled, so reuse the last saved
                    # detection result
                    dt_boxes_final = [box for box in save_dt]
                else:
                    # this frame is sampled, so use the full model result
                    dt_boxes_final = [box for box in current_full_model_dt]
                    save_dt = [box for box in dt_boxes_final]

                tp[img_index], fp[img_index], fn[img_index] = \
                        eval_single_image(current_gt, dt_boxes_final)

                # print(tp[img_index], fp[img_index],fn[img_index])
            tp_total = sum(tp.values())
            fp_total = sum(fp.values())
            fn_total = sum(fn.values())

            f1 = compute_f1(tp_total, fp_total, fn_total)
             
            print(resolution, sample_rate, f1)
            F1_score_list.append(f1)

        frame_rate_list = [standard_frame_rate/x for x in TEMPORAL_SAMPLING_LIST]
        current_f1_list = F1_score_list

        if current_f1_list[-1] < TARGET_F1:
            target_frame_rate = None
        else:
            index = next(x[0] for x in enumerate(current_f1_list) if x[1] > TARGET_F1)
            if index == 0:
                target_frame_rate = frame_rate_list[0]
            else:
                point_a = (current_f1_list[index-1], frame_rate_list[index-1])
                point_b = (current_f1_list[index], frame_rate_list[index])

                target_frame_rate  = interpolation(point_a, point_b, TARGET_F1)

        print("Resolution = {} and target frame rate = {}".format(resolution, target_frame_rate))
        result[resolution] = target_frame_rate
        # select best profile
    # good_settings = []
    min_bw = metadata['resolution'][0]*metadata['resolution'][1]*standard_frame_rate
    best_resol = 'original'
    best_frame_rate = standard_frame_rate
    for resolution in result.keys():
        target_frame_rate = result[resolution]

        if target_frame_rate == None:
            continue
        video_size = compute_video_size(dataset, start_frame, 
                                        start_frame+chunk_length*frame_rate, 
                                        target_frame_rate, frame_rate, 
                                        standard_frame_rate, resolution)
        print(resolution, video_size)
        bw = video_size

        if bw < min_bw:
            best_resol = resolution
            best_frame_rate = target_frame_rate
            min_bw = bw
    origin_bw = compute_video_size(dataset, start_frame, 
                                    start_frame+chunk_length*frame_rate, 
                                    frame_rate, frame_rate, 
                                    standard_frame_rate, 'original')
 
    return best_resol, best_frame_rate, min_bw/origin_bw


def profile_eval(dataset, frame_rate, gt, best_resolution, best_sample_rate,
        start_frame, end_frame):
    with open(PATH + dataset + '/metadata.json') as metadata_f:
        metadata = json.load(metadata_f)

    standard_frame_rate = frame_rate
    if best_resolution == 'original':
        dt_file = PATH + dataset + '/profile/updated_gt_FasterRCNN_COCO.csv' 
        image_resolution = metadata['resolution']
    else:
        dt_file = PATH + dataset + '/' + best_resolution + '/profile/updated_gt_FasterRCNN_COCO.csv'
        image_resolution = IMAGE_RESOLUTION_DICT[best_resolution]
    height = image_resolution[1]

    full_model_dt, _ = load_full_model_detection(dt_file, height)

    img_cn = 0
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    save_dt = []

    for img_index in range(start_frame, end_frame):
        dt_boxes_final = []
        current_full_model_dt = full_model_dt[img_index]
        current_gt = gt[img_index]
        resize_rate = frame_rate/standard_frame_rate
        if img_index%resize_rate >= 1:
            continue
        else:
            img_index = img_cn
            img_cn += 1


        # based on sample rate, decide whether this frame is sampled
        if img_index%best_sample_rate >= 1:
            # this frame is not sampled, so reuse the last saved
            # detection result
            dt_boxes_final = [box for box in save_dt]

        else:
            # this frame is sampled, so use the full model result
            dt_boxes_final = [box for box in current_full_model_dt]
            save_dt = [box for box in dt_boxes_final]

        tp[img_index], fp[img_index], fn[img_index] = eval_single_image(current_gt, dt_boxes_final)   
                            
    tp_total = sum(tp.values())
    fp_total = sum(fp.values())
    fn_total = sum(fn.values())

    return compute_f1(tp_total, fp_total, fn_total)

def main():

    f = open('awstream_motivation_final.csv','w')
    f.write('dataset,best_resolution,f1,frame_rate,bandwidth\n')
    for dataset in DATASET_LIST:
        with open(PATH + dataset + '/metadata.json') as metadata_f:
            metadata = json.load(metadata_f)
        height = metadata['resolution'][1]
        # load detection results of fasterRCNN + full resolution + 
        #highest frame rate as ground truth
        frame_rate = metadata['frame rate']
        standard_frame_rate = frame_rate
        gt_file = PATH + dataset + '/profile/updated_gt_FasterRCNN_COCO.csv'    
        gt, num_of_frames = load_full_model_detection(gt_file, height)  
        num_of_short_videos = num_of_frames//(SHORT_VIDEO_LENGTH*frame_rate)
        print('Processing', dataset)
        test_bw_list = list()
        test_f1_list = list()
        for i in range(num_of_short_videos):
            start_frame = i * (SHORT_VIDEO_LENGTH*frame_rate)
            end_frame = (i+1) * (SHORT_VIDEO_LENGTH*frame_rate)

            # use 30 seconds video for profiling
            chunk_length = 30
            best_resolution, best_frame_rate, best_bw = profile(dataset, frame_rate, gt, start_frame, chunk_length)
            print("Finished profiling on the {} 30s.".format(i+1))

            # test on the whole video
            best_sample_rate = standard_frame_rate / best_frame_rate

            f1 = profile_eval(dataset, frame_rate, gt, best_resolution, best_sample_rate, start_frame, end_frame)

            test_bw_list.append(best_bw)
            test_f1_list.append(f1)

            print(dataset+str(i), best_frame_rate, f1)
            f.write(dataset + '_' + str(i) + ',' + str(best_resolution) + 
                    ',' + str(f1) + ',' + str(best_frame_rate) + ',' + 
                    str(best_bw) + '\n')
        if test_bw_list and test_f1_list:
            plt.scatter(test_bw_list, test_f1_list, label=dataset)
    plt.xlabel('Bandwidth(Mbps)')
    plt.xlim(0, 1)
    plt.ylabel('F1 Score')
    plt.ylim(0,1)
    plt.title("Awstream Motivation")
    plt.legend()
    plt.savefig('/home/zxxia/figs/awstream/awstream_motivation.png')
    plt.show()

if __name__ == '__main__':
    main()
