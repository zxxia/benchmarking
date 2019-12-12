# from collections import defaultdict
from glimpse import pipeline_perfect_tracking  # , compute_target_frame_rate
# from matplotlib import pyplot as plt
from utils.model_utils import load_waymo_detection
# , load_ssd_detection
from utils.utils import load_metadata
import json
# import matplotlib
import numpy as np
import argparse

kitti = False
DATASET_LIST = ['crossroad']
DATASET_LIST = sorted(['highway', 'crossroad2', 'crossroad', 'crossroad3',
                       'crossroad4', 'driving1', 'driving2', 'traffic',
                       'jp_hw', 'russia', 'tw_road', 'tw_under_bridge', 'jp',
                       'russia1', 'highway_normal_traffic', 'tw', 'tw1',
                       'drift', 'nyc', 'motorway', 'park', 'lane_split',
                       'driving_downtown'])


# PROFILE_LENGTH = 30  # 30 seconds
# OFFSET = 0  # offset from the start of the video in unit of second
# standard_frame_rate = 10.0 # for comparison with KITTI
# PATH = '/mnt/data/zhujun/dataset/Youtube/'
# SHORT_VIDEO_LENGTH = 30
# TARGET_F1 = 0.9
PARA1_LIST = np.concatenate([np.array([2]), np.arange(5, 350, 5)])
PARA2_LIST = [0.5]
# np.concatenate([np.arange(15,1,-1), np.arange(1,0,-0.2)])#10,7,5,3,2,1,0.7]


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

    # print(args.fps, args.resolution)

    # para1_dict = {}
    # with open('.csv', 'r') as f:
    # with open('glimpse_perfect_tracking_waymo.csv', 'r') as f:
    #     f.readline()
    #     for line in f:
    #         cols = line.strip().split(',')
    #         para1_dict[cols[0]] = float(cols[1])

    # with open(
    # '/home/zxxia/benchmarking/feature_analysis/selected_videos.json',
    #           'r') as f:
    #     selected_videos = json.load(f)
    # interested_videos = []
    # for l in selected_videos.values():
    #     interested_videos.extend(l)
    # choose the first 3 mins to get the best frame diff thresh
    with open(output_file, 'a+', 1) as final_result_f:
        header = 'video chunk,para1,para2,f1,frame rate,' \
            'ideal frame rate,trigger f1\n'
        final_result_f.write(header)
        # for video_type in DATASET_LIST:
        f_profile = open('profile_perfect_tracking_true_label/profile_{}.csv'.format(video_name), 'w')
        f_trigger_log = open('profile_perfect_tracking_true_label/trigger_log_{}.json'.format(video_name), 'w')

        if metadata_file:
            metadata = load_metadata(metadata_file)
            resolution = metadata['resolution']
            fps = metadata['frame rate']
            # frame_count = metadata['frame count']
        else:
            fps = args.fps
            resolution = args.resolution

        # read ground truth and full model detection result
        # image name, detection result, ground truth

        # img_path = path + str(resolution[1]) + 'p/'
        img_path = path + '720p/'
        annot_file = img_path + 'profile/waymo_gt.csv'
        # annot_file = img_path + 'result/input_w_gt.csv'
        gt_annot, frame_end = load_waymo_detection(annot_file)
        # dt, gt_annot, img_list = load_ssd_detection(annot_file)
        # frame_end = len(img_list)
        # import pdb
        # pdb.set_trace()

        nb_short_videos = (frame_end-offset*fps)//(short_video_length*fps)

        if nb_short_videos == 0:
            nb_short_videos = 1

        for i in range(nb_short_videos):
            start = i*(short_video_length*fps)+1+offset*fps
            end = (i+1)*(short_video_length*fps)+offset*fps

            if end > frame_end:
                end = frame_end
                clip = video_name
            else:
                clip = video_name + '_' + str(i)
                # if clip not in interested_videos:
                #     continue

            profile_start = start
            profile_end = min(start+profile_length*fps-1, end)
            print("{} {} start={}, end={}".format(clip, img_path, start, end))

            # print(video_type, seg_index)
            # Run inference on the first 30s video
            min_f1_gt_target = 1.0
            # the minimum f1 score which is greater than
            # or equal to target f1(e.g. 0.9)
            min_fps = fps

            best_para1 = -1
            best_para2 = -1
            best_trigger_f1 = 0
            best_video_trigger_log = []
            # for para1 in [para1_dict[clip]]:
            for para1 in PARA1_LIST:
                para2 = PARA2_LIST[0]
                # for para2 in para2_list:
                # larger para1, smaller thresh, easier to be triggered
                frame_difference_thresh = resolution[0]*resolution[1]/para1
                tracking_error_thresh = para2
                # images start from index 1
                triggered_frame, ideal_triggered_frame, f1, \
                    trigger_f1, video_trigger_log = \
                    pipeline_perfect_tracking(img_path, gt_annot,
                                              profile_start, profile_end,
                                              frame_difference_thresh,
                                              tracking_error_thresh,
                                              img_name_format)
                current_fps = triggered_frame/float(profile_length)
                ideal_fps = ideal_triggered_frame/float(profile_length)
                print('para1 = {}, para2 = {}, '
                      'Profiled f1 = {}, Profiled perf = {}, Ideal perf={}'
                      .format(para1, para2, f1, current_fps/fps,
                              ideal_fps/fps))
                f_profile.write(','.join([clip, str(para1), str(para2),
                                          str(f1), str(current_fps/fps),
                                          str(ideal_fps/fps)])+'\n')

                if f1 >= target_f1 and \
                   f1 <= min_f1_gt_target and \
                   current_fps < min_fps:
                    # record min f1 which is greater than target f1
                    min_f1_gt_target = f1
                    min_fps = current_fps
                    # record the best config
                    best_para1 = para1
                    best_para2 = para2
                    best_trigger_f1 = trigger_f1
                    best_video_trigger_log = video_trigger_log
                    break

            print("best_para1 = {}, best_para2={}"
                  .format(best_para1, best_para2))
            # print("Start testing....")
            # use the selected parameters for the next 5 mins
            final_result_f.write(','.join([clip, str(best_para1),
                                           str(best_para2),
                                           str(min_f1_gt_target),
                                           str(min_fps/fps),
                                           str(ideal_fps/fps),
                                           str(best_trigger_f1)])+'\n')
            json.dump(best_video_trigger_log, f_trigger_log,
                      sort_keys=True, indent=4)
        f_profile.close()
        f_trigger_log.close()


if __name__ == '__main__':
    main()
