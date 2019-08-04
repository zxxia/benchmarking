from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from VideoStorm_temporal import load_full_model_detection, eval_single_image
from my_utils import interpolation, load_metadata, compute_f1

PATH  = '/mnt/data/zhujun/dataset/Youtube/'
#META_PATH  = '/mnt/data/zhujun/new_video/'
#PATH  = '/mnt/data/zhujun/new_video/'
TEMPORAL_SAMPLING_LIST = [20,15,10,5,4,3,2.5,2,1.8,1.5,1.2,1]
MODEL_LIST = ['FasterRCNN'] # MODEL_LIST = ['FasterRCNN','SSD']
DATASET_LIST = ['motorway'] 
# sorted(['traffic', 'jp_hw', 'russia', 'tw_road', 
#            'tw_under_bridge', 'highway_normal_traffic', 'nyc', 'lane_split',
#            'tw', 'tw1', 'jp', 'russia1','drift', 'park', 'walking',  'highway', 'crossroad2', 
#                  'crossroad', 'crossroad3', 'crossroad4', 'driving1', 'driving2',
#                  'motorway'])

# 'reckless_driving','motor','highway_no_traffic'

IOU_THRESH = 0.5
TARGET_F1 = 0.9
OFFSET = 0 # The time offset from the start of the video. Unit: seconds
CHUNK_LENGTH = int(2*60) # A long video is chopped into chunks. Unit: second
PROFILE_LENGTH = 30 # Profiling length within a chunk. Unit: second

def profile(dataset, frame_rate, gt, start_frame, chunk_length=30):
    result = {}
    metadata = load_metadata(PATH + dataset + '/metadata.json')
    height = metadata['resolution'][1]
    standard_frame_rate = frame_rate
    # choose model
    # choose resolution
    # choose frame rate
    for model in MODEL_LIST:
        F1_score_list = []
        if model == 'FasterRCNN':
            dt_file = PATH + dataset + '/profile/updated_gt_FasterRCNN_COCO.csv' 
        else:
            dt_file = '/home/zhujun/video_analytics_pipelines/final_code/videostorm/small_model/' + \
                                  dataset + '_updated_gt_SSD.csv'
        full_model_dt, num_of_frames = load_full_model_detection(dt_file)#, height)               
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

                tp[img_index], fp[img_index], fn[img_index] = eval_single_image(current_gt, dt_boxes_final)

                # print(tp[img_index], fp[img_index],fn[img_index])
            tp_total = sum(tp.values())
            fp_total = sum(fp.values())
            fn_total = sum(fn.values())

            f1 = compute_f1(tp_total, fp_total, fn_total)
            F1_score_list.append(f1)

        frame_rate_list = [standard_frame_rate/x for x in TEMPORAL_SAMPLING_LIST]

        current_f1_list = F1_score_list
        # print(list(zip(frame_rate_list, current_f1_list)))

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

        
        result[model] = target_frame_rate
        # select best profile
    good_settings = []
    smallest_gpu_time = 100*standard_frame_rate
    for model in result.keys():
        target_frame_rate = result[model]
        if target_frame_rate == None:
            continue
        if model == 'FasterRCNN':
            gpu_time = 100*target_frame_rate
        else:
            gpu_time = 50*target_frame_rate
        
        if gpu_time < smallest_gpu_time:
            best_model = model
            best_frame_rate = target_frame_rate

    return best_model, best_frame_rate


def profile_eval(dataset, frame_rate, gt, best_model, best_sample_rate, 
                 start_frame, end_frame):
    metadata = load_metadata(PATH + dataset + "/metadata.json")
    height = metadata['resolution'][1]
    standard_frame_rate = frame_rate

    if best_model == 'FasterRCNN':
        dt_file = PATH + dataset + '/profile/updated_gt_FasterRCNN_COCO.csv' 
    else:
        dt_file = '/home/zhujun/video_analytics_pipelines/final_code/videostorm/small_model/' + \
                              dataset + '_updated_gt_SSD.csv'       
    full_model_dt, _ = load_full_model_detection(dt_file)#, height)

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

    f1 = compute_f1(tp_total, fp_total, fn_total)
    return f1

def main():

    # short_video_length = 5*60 # divide each video into 5-min
    with open('videostorm_motivation_result_motorway.csv', 'w') as f:
        f.write("video_name,frame_rate,f1\n")
        for dataset in DATASET_LIST:
            print("processing", dataset)
            metadata = load_metadata(PATH + dataset + '/metadata.json')
            height = metadata['resolution'][1]
            frame_rate = metadata['frame rate']
            standard_frame_rate = frame_rate
            
            
            # load fasterRCNN + full resolution + highest frame rate as ground truth
            gt_file = PATH + dataset + '/profile/updated_gt_FasterRCNN_COCO.csv'    
            gt, num_of_frames = load_full_model_detection(gt_file)#, height)  
           
            # Chop long videos into small chunks
            # Floor division drops the last sequence of frames which is not as long as CHUNK_LENGTH
            num_of_chunks = (num_of_frames - OFFSET * frame_rate)// (CHUNK_LENGTH * frame_rate)
            
            
            test_f1_list = list()
            test_fps_list = list()

            for i in range(num_of_chunks): 
                # the 1st frame in the chunk
                start_frame = i * (CHUNK_LENGTH * frame_rate) + 1 + OFFSET * frame_rate
                # the last frame in teh chunk
                end_frame = (i+1) * (CHUNK_LENGTH * frame_rate) + OFFSET * frame_rate
                print('short video start={}, end={}'.format(start_frame, end_frame)) 
                # profile the first PROFILE_LENGTH seconds of the chunk
                assert(CHUNK_LENGTH > PROFILE_LENGTH)
                best_model, best_frame_rate = profile(dataset, frame_rate, gt, 
                                                      start_frame, PROFILE_LENGTH)
                # test on the whole video
                best_sample_rate = standard_frame_rate / best_frame_rate

                f1 = profile_eval(dataset, frame_rate, gt, best_model, 
                                  best_sample_rate, start_frame+PROFILE_LENGTH * frame_rate, end_frame)

                test_f1_list.append(f1)
                test_fps_list.append(best_frame_rate)
                print(dataset+str(i), best_frame_rate, f1)
                f.write(dataset + '_' + str(i) + ',' + 
                        str(best_frame_rate/standard_frame_rate) + ','+ str(f1)+ '\n')
            test_relative_fps_list = [x/standard_frame_rate for x in test_fps_list]
            #if test_relative_fps_list and test_f1_list:
            #    plt.scatter(test_relative_fps_list, test_f1_list, label=dataset)
    # plt.xlabel("GPU Processing Time")
    # plt.ylabel("Accuracy")
    # plt.legend()
    # plt.xlim(0,1)
    # plt.ylim(0,1)
    # plt.title("VideoStorm Motivation")
    # plt.savefig("/home/zxxia/figs/videostorm/videostorm_motivation.png")
    # plt.show()


if __name__ == '__main__':
    main()
