from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from utils.model_utils import load_waymo_detection, load_full_model_detection
from utils.utils import load_metadata
from videostorm.profiler import profile, profile_eval
import glob
import re

PATH = '/mnt/data/zhujun/new_video/'
TEMPORAL_SAMPLING_LIST = [20, 15, 10, 5, 4, 3, 2.5, 2, 1.8, 1.5, 1.2, 1]
MODEL_LIST = ['FasterRCNN']  # MODEL_LIST = ['FasterRCNN','SSD']
DATASET_LIST = ['training_0000', 'training_0001', 'training_0002',
                'validation_0000']
CAMERAS = ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT']
TARGET_F1 = 0.9
OFFSET = 0  # The time offset from the start of the video. Unit: seconds
CHUNK_LENGTH = int(2*60)  # A long video is chopped into chunks. Unit: second
PROFILE_LENGTH = 6  # Profiling length within a chunk. Unit: second
WAYMO_FRONT_CAMERA_FRAME_RATE = 10
WAYMO_FRONT_CAMERA_RESOLUTION = [1920, 1280]


def main():
    with open('videostorm_waymo_coverage.csv', 'w') as f:
        f.write("video_name,frame_rate,f1\n")
        for dataset in DATASET_LIST:
            print("processing", dataset)
            segment_paths = sorted(glob.glob(PATH+dataset+'/segment-*/'))
            for segment_path in segment_paths:
                rz = re.findall(".*(segment-.*)/", segment_path)
                segment_name = rz[0]
                print(rz)
                # import pdb
                # pdb.set_trace()
                for camera in CAMERAS:
                    clip = dataset+'_'+segment_name+'_'+camera
                    metadata = load_metadata(segment_path+'/metadata.json')
                    print(segment_path + camera)
                    frame_rate = metadata['frame rate']
                    # resolution = metadata['resolution']
                    dt_file = segment_path + '/' + camera + '/720p/'+'profile/updated_gt_FasterRCNN_COCO.csv'
                    # gt, num_of_frames = load_waymo_detection(dt_file)
                    gt, num_of_frames = load_full_model_detection(dt_file)
                    # print(dt_file, num_of_frames)

            #     rz = re.findall("gt_(.*).csv", gt_file)
            #     print(rz[0])

            #     # Chop long videos into small chunks
            #     # Floor division drops the last sequence of frames which
            #     # is not as  long as CHUNK_LENGTH
                    profile_frame_cnt = PROFILE_LENGTH * frame_rate
                    chunk_frame_cnt = CHUNK_LENGTH * frame_rate
                    chunk_frame_cnt = num_of_frames
                    num_of_chunks = (num_of_frames - OFFSET * frame_rate) \
                        // chunk_frame_cnt

                    test_f1_list = list()
                    test_fps_list = list()
                    for i in range(num_of_chunks):
                        # the 1st frame in the chunk
                        start_frame = i * chunk_frame_cnt+1+OFFSET*frame_rate
                        # the last frame in the chunk
                        end_frame = (i + 1) * chunk_frame_cnt + OFFSET * frame_rate
                        print('short video start={}, end={}'.format(start_frame,
                                                                    end_frame))
                        # profile the first PROFILE_LENGTH seconds of the chunk
                        assert(CHUNK_LENGTH >= PROFILE_LENGTH)

                        profile_start_frame = start_frame
                        profile_end_frame = profile_start_frame + profile_frame_cnt - 1
                        print('profile short video start={}, end={}'.format(profile_start_frame,
                                                                    profile_end_frame))

                        best_frame_rate = profile(gt, gt, profile_start_frame,
                                                  profile_end_frame, frame_rate,
                                                  TEMPORAL_SAMPLING_LIST)
                        # test on the rest of the short video
                        best_sample_rate = frame_rate/best_frame_rate

                        test_start_frame = profile_end_frame + 1
                        test_end_frame = end_frame

                        print('test short video start={}, end={}'.format(test_start_frame,
                                                                    test_end_frame))
                        f1 = profile_eval(gt, gt, frame_rate, best_sample_rate,
                                          test_start_frame, test_end_frame)

                        test_f1_list.append(f1)
                        test_fps_list.append(best_frame_rate)
                        print(clip, best_frame_rate/frame_rate, f1)
                        f.write(','.join([clip, str(best_frame_rate/frame_rate), str(f1) + '\n']))


if __name__ == '__main__':
    main()
