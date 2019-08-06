import pdb
from videostorm.profiler import profile, profile_eval
from utils.model_utils import load_fastrcnn_detection
from utils.utils import load_metadata

PATH  = '/mnt/data/zhujun/dataset/Youtube/XIPH/'
TEMPORAL_SAMPLING_LIST = [20,15,10,5,4,3,2.5,2,1.8,1.5,1.2,1]
MODEL_LIST = ['FasterRCNN'] # MODEL_LIST = ['FasterRCNN','SSD']

TARGET_F1 = 0.9
OFFSET = 0 # The time offset from the start of the video. Unit: seconds
CHUNK_LENGTH = int(2*60) # A long video is chopped into chunks. Unit: second
PROFILE_LENGTH = 30 # Profiling length within a chunk. Unit: second
KITTI_FRAME_RATE = 10
DATASET_LIST = sorted(['akiyo_cif', 'deadline_cif', 'ice_4cif', 'crew_4cif', 
                       'highway_cif', 'pedestrian_area_1080p25', 'bowing_cif', 
                       'football_422_ntsc', 'KristenAndSara_1280x720_60', 
                       'rush_hour_1080p25', 'bus_cif', 'foreman_cif', 
                       'mad900_cif', 'station2_1080p25', 'carphone_qcif', 
                       'FourPeople_1280x720_60', 'miss_am_qcif', 
                       'tractor_1080p25', 'claire_qcif', 'grandma_qcif', 
                       'mthr_dotr_qcif', 'coastguard_cif', 'hall_objects_qcif',
                       'Netflix_Crosswalk_4096x2160_60fps_10bit_420', 
                       'Netflix_DrivingPOV_4096x2160_60fps_10bit_420'])

def main():
    with open('videostorm_xiph_coverage.csv', 'w') as f:
        f.write("video_name,frame_rate,f1\n")
        for dataset in DATASET_LIST:
            print("processing", dataset)
            metadata = load_metadata(PATH + dataset + '/metadata.json')
            frame_rate = metadata['frame rate']
            frame_cnt = metadata['frame count']
            # profiling 

            gt_file = PATH + dataset + '/profile/updated_gt_FasterRCNN_COCO.csv'
            gt, _ = load_fastrcnn_detection(gt_file)
            print(gt_file) 
            
            profile_start_frame = 1
            profile_end_frame = int(0.2 * frame_cnt)
            
            best_frame_rate = profile(gt, gt, profile_start_frame, 
                                      profile_end_frame, frame_rate, 
                                      TEMPORAL_SAMPLING_LIST)
            print(best_frame_rate)

            test_f1_list = list()
            test_fps_list = list()


            # test on the rest of the short video
            test_start_frame = profile_end_frame + 1
            test_end_frame = frame_cnt
            best_sample_rate = frame_rate / best_frame_rate
            f1 = profile_eval(gt, gt, frame_rate, best_sample_rate, 
                              test_start_frame, test_end_frame)

            test_f1_list.append(f1)
            test_fps_list.append(best_frame_rate)
            print(dataset, best_frame_rate, f1)
            f.write(dataset + ',' + 
                    str(1/best_sample_rate) + ',' + str(f1) + '\n')
            test_relative_fps_list = [x/frame_rate for x in test_fps_list]

if __name__ == '__main__':
    main()
