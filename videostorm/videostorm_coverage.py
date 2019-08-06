import pdb
from videostorm.profiler import profile, profile_eval
from utils.model_utils import load_ssd_detection
PATH  = '/mnt/data/zhujun/dataset/KITTI/'
TEMPORAL_SAMPLING_LIST = [20,15,10,5,4,3,2.5,2,1.8,1.5,1.2,1]
MODEL_LIST = ['FasterRCNN'] # MODEL_LIST = ['FasterRCNN','SSD']
DATASET_LIST = ['City', 'Residential', 'Road']

TARGET_F1 = 0.9
OFFSET = 0 # The time offset from the start of the video. Unit: seconds
CHUNK_LENGTH = int(2*60) # A long video is chopped into chunks. Unit: second
PROFILE_LENGTH = 30 # Profiling length within a chunk. Unit: second
KITTI_FRAME_RATE = 10
VIDEO_INDEX_DICT = {'City':[1,2,5,9,11,13,14,17,18,48,51,56,57,59,60,84,91,93],
                    'Road':[15,27,28,29,32,52,70],
                    'Residential':[19,20,22,23,35,36,39,46,61,64,79,86,87]}

def main():
    with open('videostorm_motivation_result_kitti.csv', 'w') as f:
        f.write("video_name,frame_rate,f1\n")
        for dataset in DATASET_LIST:
            print("processing", dataset)
            frame_rate = KITTI_FRAME_RATE
            # profiling 
            if dataset == 'City':
                profiling_video_idx = 93
            elif dataset == 'Residential':
                profiling_video_idx = 19
            else:
                profiling_video_idx = 28

            
            # Test stage
            gt_file = PATH + dataset + '/2011_09_26_drive_' + \
                      format(profiling_video_idx, '04d') + \
                      '_sync/result/input_w_gt.csv'
            dt, gt, img_list = load_ssd_detection(gt_file)
            print(gt_file) 

            
            best_frame_rate = profile(gt, dt, img_list[0], img_list[-1], 
                                      frame_rate, TEMPORAL_SAMPLING_LIST)
            print(best_frame_rate)

            test_f1_list = list()
            test_fps_list = list()

            # load fasterRCNN + full resolution + highest frame rate as ground truth
            for i, video_idx in enumerate(VIDEO_INDEX_DICT[dataset]):
                if video_idx == profiling_video_idx:
                    continue
                gt_file = PATH + dataset + '/2011_09_26_drive_' + \
                          format(video_idx, '04d') + \
                          '_sync/result/input_w_gt.csv'
                dt, gt, img_list = load_ssd_detection(gt_file)
                print(gt_file) 

                # test on the rest of the short video
                best_sample_rate = frame_rate / best_frame_rate
                f1 = profile_eval(gt, dt, frame_rate, best_sample_rate, 
                                  img_list[0], img_list[-1])

                test_f1_list.append(f1)
                test_fps_list.append(best_frame_rate)
                print(dataset+str(video_idx), best_frame_rate, f1)
                f.write(dataset + '_' + str(video_idx) + ',' + 
                        str(1/best_sample_rate) + ',' + str(f1) + '\n')
                test_relative_fps_list = [x/frame_rate for x in test_fps_list]

if __name__ == '__main__':
    main() 
