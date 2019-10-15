""" VideoStorm Overfitting Script """
import argparse
from utils.model_utils import load_full_model_detection, \
    filter_video_detections
from utils.utils import load_metadata
from videostorm.profiler import profile, profile_eval
from constants import CAMERA_TYPES
# PATH = '/mnt/data/zhujun/dataset/Youtube/'
PATH = '/data/zxxia/videos/'
CSV_PATH = '/data/zxxia/benchmarking/results/videos/'
TEMPORAL_SAMPLING_LIST = [20, 15, 10, 5, 4, 3, 2.5, 2, 1.8, 1.5, 1.2, 1]
# DATASET_LIST = sorted(['crossroad', 'crossroad2', 'crossroad3', 'crossroad4',
#                        'drift', 'driving1', 'driving2', 'driving_downtown',
#                        'highway', 'highway_normal_traffic', 'jp', 'jp_hw',
#                        'lane_split', 'motorway', 'nyc', 'park', 'russia',
#                        'russia1', 'traffic', 'tw', 'tw1', 'tw_road',
#                        'tw_under_bridge'])

# 'reckless_driving','motor','highway_no_traffic' 'crossroad'
TARGET_F1 = 0.9
OFFSET = 0  # The time offset from the start of the video. Unit: seconds
CHUNK_LENGTH = 30  # A long video is chopped into chunks. Unit: second
PROFILE_LENGTH = 30  # Profiling length within a chunk. Unit: second


def main():
    parser = argparse.ArgumentParser(
        description="VideoStorm with temporal overfitting")
    parser.add_argument("--video", type=str, required=True, help="video name")
    parser.add_argument("--input", type=str, required=True,
                        help="input full model detection file")
    parser.add_argument("--output", type=str, required=True,
                        help="output result file")
    parser.add_argument("--log", type=str, required=True, help="log file")
    args = parser.parse_args()
    dataset = args.video
    output_file = args.output
    input_file = args.input
    log_file = args.log
    f_log = open(log_file, 'w', 1)
    f_log.write("video_name,frame_rate,f1\n")
    with open(output_file, 'w', 1) as f_out:
        f_out.write("video_name,frame_rate,f1\n")
        print("processing", dataset)
        metadata = load_metadata(PATH + dataset + '/metadata.json')
        # height = metadata['resolution'][1]
        frame_rate = metadata['frame rate']
        # resolution = metadata['resolution']
        # resol_dir = str(resolution[1]) + 'p'
        # resol_dir = '720p'

        gtruth, num_of_frames = load_full_model_detection(input_file)

        gtruth = filter_video_detections(gtruth, target_types={3, 8})
        # Filter ground truth if it is static camera
        if dataset in CAMERA_TYPES['static']:
            gtruth = filter_video_detections(gtruth, width_range=(0, 1280/2),
                                             height_range=(0, 720/2))
        gtruth = filter_video_detections(gtruth, height_range=(720//20, 720))

        # Chop long videos into small chunks
        # Floor division drops the last sequence of frames which is not as
        # long as CHUNK_LENGTH
        profile_frame_cnt = PROFILE_LENGTH * frame_rate
        chunk_frame_cnt = CHUNK_LENGTH * frame_rate
        num_of_chunks = (num_of_frames-OFFSET*frame_rate)//chunk_frame_cnt

        test_f1_list = list()
        test_fps_list = list()
        for i in range(num_of_chunks):
            clip = dataset + '_' + str(i)
            # the 1st frame in the chunk
            start_frame = i * chunk_frame_cnt + 1 + OFFSET * frame_rate
            # the last frame in the chunk
            end_frame = (i + 1) * chunk_frame_cnt + OFFSET * frame_rate
            print('short video start={}, end={}'.format(start_frame,
                                                        end_frame))
            # profile the first PROFILE_LENGTH seconds of the chunk
            assert CHUNK_LENGTH >= PROFILE_LENGTH

            profile_start_frame = start_frame
            profile_end_frame = profile_start_frame+profile_frame_cnt-1

            best_frame_rate = profile(clip, gtruth, gtruth,
                                      profile_start_frame, profile_end_frame,
                                      frame_rate, TEMPORAL_SAMPLING_LIST,
                                      f_log)
            # test on the rest of the short video
            best_sample_rate = frame_rate/best_frame_rate

            test_start_frame = profile_start_frame
            test_end_frame = profile_end_frame

            f1_score = profile_eval(gtruth, gtruth, best_sample_rate,
                                    test_start_frame, test_end_frame)

            test_f1_list.append(f1_score)
            test_fps_list.append(best_frame_rate)
            print(dataset+str(i), best_frame_rate, f1_score)
            f_out.write(','.join([clip, str(best_frame_rate/frame_rate),
                                  str(f1_score)])+'\n')
    f_log.close()


if __name__ == '__main__':
    main()
