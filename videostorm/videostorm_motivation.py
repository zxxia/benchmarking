from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from utils.model_utils import load_full_model_detection
from utils.utils import load_metadata
from profiler import profile, profile_eval
PATH  = '/mnt/data/zhujun/dataset/Youtube/'
TEMPORAL_SAMPLING_LIST = [20,15,10,5,4,3,2.5,2,1.8,1.5,1.2,1]
MODEL_LIST = ['FasterRCNN'] # MODEL_LIST = ['FasterRCNN','SSD']
DATASET_LIST = ['motorway'] 
# sorted(['traffic', 'jp_hw', 'russia', 'tw_road', 
#            'tw_under_bridge', 'highway_normal_traffic', 'nyc', 'lane_split',
#            'tw', 'tw1', 'jp', 'russia1','drift', 'park', 'walking',  'highway', 'crossroad2', 
#                  'crossroad', 'crossroad3', 'crossroad4', 'driving1', 'driving2',
#                  'motorway'])

# 'reckless_driving','motor','highway_no_traffic'
TARGET_F1 = 0.9
OFFSET = 0 # The time offset from the start of the video. Unit: seconds
CHUNK_LENGTH = int(2*60) # A long video is chopped into chunks. Unit: second
PROFILE_LENGTH = 30 # Profiling length within a chunk. Unit: second

def main():
    with open('videostorm_motivation_result_test.csv', 'w') as f:
        f.write("video_name,frame_rate,f1\n")
        for dataset in DATASET_LIST:
            print("processing", dataset)
            metadata = load_metadata(PATH + dataset + '/metadata.json')
            # height = metadata['resolution'][1]
            frame_rate = metadata['frame rate']
            
            # load fasterRCNN + full resolution + highest frame rate as ground truth
            gt_file = PATH + dataset + '/profile/updated_gt_FasterRCNN_COCO.csv'    
            gt, num_of_frames = load_full_model_detection(gt_file)
           
            # Chop long videos into small chunks
            # Floor division drops the last sequence of frames which is not as
            # long as CHUNK_LENGTH
            profile_frame_cnt = PROFILE_LENGTH * frame_rate
            chunk_frame_cnt = CHUNK_LENGTH * frame_rate
            num_of_chunks = (num_of_frames - OFFSET * frame_rate)//chunk_frame_cnt
            
            test_f1_list = list()
            test_fps_list = list()
            for i in range(num_of_chunks): 
                # the 1st frame in the chunk
                start_frame = i * chunk_frame_cnt + 1 + OFFSET * frame_rate
                # the last frame in the chunk
                end_frame = (i + 1) * chunk_frame_cnt + OFFSET * frame_rate
                print('short video start={}, end={}'.format(start_frame, 
                                                            end_frame)) 
                # profile the first PROFILE_LENGTH seconds of the chunk
                assert(CHUNK_LENGTH > PROFILE_LENGTH)
            
                profile_start_frame = start_frame
                profile_end_frame = profile_start_frame + profile_frame_cnt

                best_model, best_frame_rate = profile(PATH, dataset, frame_rate, gt, 
                                                      profile_start_frame, 
                                                      profile_end_frame,
                                                      MODEL_LIST,
                                                      TEMPORAL_SAMPLING_LIST)
                # test on the rest of the short video
                best_sample_rate = frame_rate / best_frame_rate

                test_start_frame = profile_end_frame + 1
                test_end_frame = end_frame

                f1 = profile_eval(PATH, dataset, frame_rate, gt, best_model, 
                                  best_sample_rate, test_start_frame, 
                                  test_end_frame)

                test_f1_list.append(f1)
                test_fps_list.append(best_frame_rate)
                print(dataset+str(i), best_frame_rate, f1)
                f.write(dataset + '_' + str(i) + ',' + 
                        str(best_frame_rate/frame_rate) + ','+ str(f1)+ '\n')
            test_relative_fps_list = [x/frame_rate for x in test_fps_list]
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
