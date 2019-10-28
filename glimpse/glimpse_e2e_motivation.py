# import matplotlib
# matplotlib.use('Agg')
from glimpse import pipeline, compute_target_frame_rate
# from matplotlib import pyplot as plt
from utils.model_utils import load_full_model_detection
from utils.utils import load_metadata
import argparse
import json
import numpy as np
display = False
# kitti = False
# DATASET_LIST = sorted(['highway', 'crossroad2', 'crossroad', 'crossroad3',
#                        'crossroad4', 'driving1','driving2','traffic','jp_hw',
#                        'russia', 'tw_road', 'tw_under_bridge','jp','russia1',
#                        'highway_normal_traffic', 'tw', 'tw1', 'drift', 'nyc',
#                        'motorway', 'park', 'lane_split', 'driving_downtown'])

# PROFILE_LENGTH = 30  # 30 seconds
# OFFSET = 0  # offset from the start of the video in unit of second
# # standard_frame_rate = 10.0 # for comparison with KITTI
# PATH = '/mnt/data/zhujun/dataset/Youtube/'
# SHORT_VIDEO_LENGTH = 30
# INFERENCE_TIME = 100 # avg. GPU processing  time
# TARGET_F1 = 0.9

# drift parameters
# PARA1_LIST = [100, 150, 180, 200, 220, 250]
# PARA2_LIST = np.concatenate([np.arange(6, 1, -2), np.arange(1, 0, -0.2)])

# russia1 parameters
PARA1_LIST = [100, 120, 140, 150, 160, 170]
PARA1_DICT = {'crossroad': np.arange(20, 70, 10),
              'crossroad2': np.arange(20, 70, 10),
              'crossroad3': np.arange(50, 70, 10),
              'crossroad4': np.arange(30, 80, 10),
              'drift': np.arange(150, 350, 50),
              'driving1': np.arange(20, 70, 10),
              'driving2': np.arange(20, 100, 20),
              'driving_downtown': np.arange(10, 50, 10),
              'highway': np.arange(20, 50, 10),
              'jp': np.arange(20, 50, 10),
              'lane_split': np.arange(10, 20, 5),
              'motorway': np.arange(2, 8, 2),
              'nyc': np.arange(10, 50, 10),
              'park': np.arange(2, 10, 5),
              'russia': np.arange(100, 200, 50),
              'russia1': np.arange(50, 250, 50),
              'traffic': np.arange(20, 50, 5),
              'tw': np.arange(10, 50, 10),
              'tw1': np.arange(20, 70, 10),
              'tw_road': np.arange(20, 50, 5),
              'tw_under_bridge': np.arange(100, 200, 40)}
PARA2_LIST = np.arange(1.0, 0, -0.2)


def main():
    parser = argparse.ArgumentParser(
        description="Glimpse with perfect tracking")
    parser.add_argument("--path", type=str,
                        help="path contains all datasets")
    parser.add_argument("--video", type=str,
                        help="video name")
    parser.add_argument("--metadata", type=str, default='',
                        help="metadata file in Json")
    parser.add_argument("--output", type=str,
                        help="output result file")
    parser.add_argument("--short_video_length", type=int,
                        help="short video length in seconds")
    parser.add_argument("--profile_length", type=int,
                        help="profile length in seconds")
    parser.add_argument("--offset", type=int,
                        help="offset from beginning of the video in seconds")
    parser.add_argument("--target_f1", type=float, help="target F1 score")
    parser.add_argument("--fps", type=int, default=0, help="frame rate")
    parser.add_argument("--resolution", nargs='+', type=int,
                        default=[], action='store', help="video resolution")
    parser.add_argument("--format", type=str, default='{:06d}.jpg',
                        help="image name format")

    args = parser.parse_args()
    path = args.path
    video_name = args.video
    output_file = args.output
    metadata_file = args.metadata
    short_video_length = args.short_video_length
    profile_length = args.profile_length
    offset = args.offset
    target_f1 = args.target_f1
    img_name_format = args.format

    # para1_dict = {}
    # with open('glimpse_perfect_tracking.csv', 'r') as f:
    #     f.readline()
    #     for line in f:
    #         cols = line.strip().split(',')
    #         para1_dict[cols[0]] = float(cols[1])

    # with open(
    #     '/home/zxxia/benchmarking/feature_analysis/selected_videos.json',
    #         'r') as f:
    #     selected_videos = json.load(f)
    # interested_videos = []
    # for l in selected_videos.values():
    #     interested_videos.extend(l)
    # choose the first 3 mins to get the best frame diff thresh
    with open(output_file, 'w', 1) as final_result_f:
        header = 'video,para1,para2,f1,gpu,ideal frame rate,trigger f1\n'
        final_result_f.write(header)
        # for video_type in DATASET_LIST:
        f_profile = open('tmp_profile_E2E/profile_{}.csv'.format(video_name), 'w')
        metadata = load_metadata(metadata_file)
        resolution = metadata['resolution']
        fps = metadata['frame rate']
        # frame_count = metadata['frame count']

        # resize the current video frames to 10Hz
        # read ground truth and full model detection result
        # image name, detection result, ground truth

        img_path = path + str(resolution[1]) + 'p/'
        # img_path = path + '720p/'
        # f_video_log = open('tmp_profile_E2E/profile_{}.json'.format(video_name),
                           # 'w')
        annot_file = img_path + 'profile/updated_gt_FasterRCNN_COCO.csv'
        gt_annot, frame_end = load_full_model_detection(annot_file)

        num_of_short_videos = (frame_end-offset*fps)//(short_video_length*fps)

        # video_log = {}
        for i in range(num_of_short_videos):
            start = i * (short_video_length*fps) + 1 + offset * fps
            end = (i+1)*(short_video_length*fps) + offset * fps
            clip = video_name + '_' + str(i)
            # if clip not in interested_videos:
            #     continue

            profile_start = start
            profile_end = min(start+profile_length*fps-1, end)
            test_start = profile_end + 1
            test_end = end
            print("profile {} {} start={}, end={}"
                  .format(clip, img_path, profile_start, profile_end))
            # print(video_type, seg_index)
            # Run inference on the first 30s video
            min_f1_gt_target = 1.0
            # the minimum f1 score which is greater than
            # or equal to target f1(e.g. 0.9)
            min_fps = fps

            best_para1 = -1
            best_para2 = -1
            # best_para1 = 150
            # best_para2 = 0.8
            best_trigger_f1 = 0
            profiling_logs = list()

            fps_list = list()
            f1_list = list()
            # for para1 in [para1_dict[clip]]:
            # for para1 in PARA1_LIST:
            for para1 in PARA1_DICT[video_name]:
                for para2 in PARA2_LIST:
                    # larger para1, smaller thresh, easier to be triggered
                    frame_diff_th = resolution[0]*resolution[1]/para1
                    tracking_err_th = para2
                    # images start from index 1
                    triggered_frame, ideal_triggered_frame, f1, trigger_f1, \
                        frames_log = pipeline(img_path, gt_annot,
                                              profile_start, profile_end,
                                              resolution, frame_diff_th,
                                              tracking_err_th, img_name_format,
                                              display)

                    current_fps = triggered_frame/profile_length
                    ideal_fps = ideal_triggered_frame/profile_length
                    profiling_log = {'para1': para1,
                                     'para2': para2,
                                     'frame difference threshold':
                                     float(frame_diff_th),
                                     'tracking error threshold':
                                     float(tracking_err_th),
                                     'frames log': frames_log,
                                     'processing time': float(current_fps/fps),
                                     'ideal processing time':
                                     float(ideal_fps/fps),
                                     'f1': float(f1),
                                     'frame difference trigger f1':
                                     float(trigger_f1)}
                    profiling_logs.append(profiling_log)
                    print('para1 = {}, para2 = {}, '
                          'Profiled f1 = {}, Profiled perf = {}, Ideal perf={}'
                          .format(para1, para2, f1, current_fps/fps,
                                  ideal_fps/fps))
                    f_profile.write(','.join([clip, str(para1), str(para2),
                                              str(f1), str(current_fps/fps),
                                              str(ideal_fps/fps)])+'\n')
                    fps_list.append(current_fps)
                    f1_list.append(f1)
                    if f1 >= target_f1:
                        break

                if f1 >= target_f1 \
                   and f1 <= min_f1_gt_target \
                   and current_fps < min_fps:
                    min_f1_gt_target = f1
                    min_fps = current_fps
                    # record the best config
                    best_para1 = para1
                    best_para2 = para2
                    best_trigger_f1 = trigger_f1
                    break
            # video_log[clip] = profiling_logs
            print("best_para1 = {}, best_para2={}"
                  .format(best_para1, best_para2))

            frame_diff_th = resolution[0]*resolution[1]/best_para1
            tracking_err_th = best_para2
            triggered_frame, ideal_triggered_frame, f1, trigger_f1, \
                frames_log = pipeline(img_path, gt_annot, test_start, test_end,
                                      resolution, frame_diff_th,
                                      tracking_err_th, img_name_format,
                                      display)

            print("test {} {} start={}, end={}"
                  .format(clip, img_path, test_start, test_end))
            current_fps = triggered_frame/profile_length
            ideal_fps = ideal_triggered_frame/profile_length
            final_result_f.write(','.join([clip, str(best_para1),
                                           str(best_para2),
                                           str(f1),
                                           str(current_fps/fps),
                                           str(ideal_fps/fps),
                                           str(trigger_f1)])+'\n')
            # json.dump(video_log, f_video_log, sort_keys=True, indent=4)

        f_profile.close()
        # f_video_log.close()


if __name__ == '__main__':
    main()
