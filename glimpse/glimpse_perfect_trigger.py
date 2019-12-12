from collections import defaultdict
from glimpse import pipeline_perfect_trigger  # , compute_target_frame_rate
# import matplotlib
# matplotlib.use('Agg')
# from matplotlib import pyplot as plt
from utils.utils import load_metadata
from utils.model_utils import load_full_model_detection
import numpy as np
import json

kitti = False
DATASET_LIST = ['highway']
# DATASET_LIST = sorted(['highway', 'crossroad2', 'crossroad', 'crossroad3',
                       # 'crossroad4', 'driving1', 'driving2', 'traffic',
                       # 'jp_hw', 'russia', 'tw_road', 'tw_under_bridge', 'jp',
                       # 'russia1', 'highway_normal_traffic', 'tw', 'tw1', 'nyc',
                       # 'drift',  'motorway', 'park', 'lane_split',
                       # 'driving_downtown'])

PROFILE_LENGTH = 30  # 30 seconds
OFFSET = 0  # offset from the start of the video in unit of second
# standard_frame_rate = 10.0 # for comparison with KITTI
PATH = '/mnt/data/zhujun/dataset/Youtube/'
SHORT_VIDEO_LENGTH = 30
INFERENCE_TIME = 100  # avg. GPU processing  time
TARGET_F1 = 0.9
para1_list = [1]  # [2,3,4,5,6,8,10,15] #[1,2,5,10]#
para2_list = np.concatenate([np.arange(20, 1, -5), np.arange(1, 0, -0.2)])
# 10,7,5,3,2,1,0.7] #[0.1, 1, 2 ,5, 10]


def main():
    with open('/home/zxxia/benchmarking/feature_analysis/selected_videos.json',
              'r') as f:
        selected_videos = json.load(f)

    interested_videos = []
    for l in selected_videos.values():
        interested_videos.extend(l)
    # print(interested_videos)

    # choose the first 3 mins to get the best frame diff thresh
    # with open('perfect_triggering.csv', 'w', 1) as final_result_f:
    with open('test4.csv', 'w', 1) as final_result_f:
        final_result_f.write('video chunk,para1,para2,f1,frame rate\n')
        for video_type in DATASET_LIST:
            f_profile = open('tracking_{}.csv'.format(video_type), 'w')
            metadata_file = PATH + video_type + '/metadata.json'
            metadata = load_metadata(metadata_file)
            resolution = metadata['resolution']
            frame_rate = metadata['frame rate']
            # frame_count = metadata['frame count']

            # read ground truth and full model detection result
            # image name, detection result, ground truth
            annot_path = PATH + video_type + '/' + str(resolution[1]) + \
                'p/profile/updated_gt_FasterRCNN_COCO.csv'
            img_path = PATH + video_type + '/' + str(resolution[1]) + 'p/'
            gt_annot, frame_end = load_full_model_detection(annot_path)
            dt_annot, frame_end = load_full_model_detection(annot_path)

            num_of_short_videos = (frame_end-OFFSET*frame_rate) // \
                (SHORT_VIDEO_LENGTH*frame_rate)

            for i in range(num_of_short_videos):
                start = i*(SHORT_VIDEO_LENGTH*frame_rate)+1+OFFSET*frame_rate
                end = (i+1)*(SHORT_VIDEO_LENGTH*frame_rate)+OFFSET*frame_rate

                clip = video_type + '_' + str(i)
                if clip not in interested_videos:
                    continue

                profile_start = start
                profile_end = start + PROFILE_LENGTH * frame_rate - 1
                print("short video start={}, end={}".format(start, end))
                print("profile start={}, end={}".format(profile_start,
                                                        profile_end))
                print('profiling short video {}'.format(i))

                # Run inference on the first 30s video
                # the minimum f1 score which is greater than or equal to
                # target f1(e.g. 0.9)
                min_f1_gt_target = 1.0
                min_fps = frame_rate

                best_para1 = -1
                best_para2 = -1
                para1 = para1_list[0]
                for para2 in para2_list:
                    # csvf = open('no_meaning.csv', 'w')
                    # larger para1, smaller thresh, easier to be triggered
                    frame_diff_th = resolution[0]*resolution[1]/para1
                    tracking_err_th = para2
                    # images start from index 1
                    print(img_path, profile_start, profile_end, para1, para2)
                    triggered_frame, ideal_triggered_frame, f1 = \
                        pipeline_perfect_trigger(img_path, dt_annot, gt_annot,
                                                 profile_start, profile_end,
                                                 resolution,  frame_diff_th,
                                                 tracking_err_th, False)

                    current_fps = triggered_frame / float(PROFILE_LENGTH)
                    ideal_fps = ideal_triggered_frame/float(PROFILE_LENGTH)
                    # frame_rate_list.append(current_fps)
                    # f1_list.append(f1)
                    print('para1 = {}, para2 = {}, Profiled f1 = {}, '
                          'Profiled gpu = {}'
                          .format(para1, para2, f1, current_fps/frame_rate))
                    f_profile.write(','.join([video_type + '_' + str(i),
                                              str(para1), str(para2), str(f1),
                                              str(current_fps/frame_rate),
                                              str(ideal_fps/frame_rate)])+'\n')
                    if f1 >= TARGET_F1:
                        break

                if f1 >= TARGET_F1 and \
                   f1 <= min_f1_gt_target and \
                   current_fps < min_fps:
                    # record min f1 which is greater than target f1
                    min_f1_gt_target = f1
                    min_fps = current_fps
                    # record the best config
                    best_para1 = para1
                    best_para2 = para2

                final_result_f.write(','.join([clip, str(best_para1),
                                               str(best_para2), str(f1),
                                               str(min_fps/frame_rate),
                                               str(ideal_fps/frame_rate)])
                                     + '\n')
                # break
            f_profile.close()


if __name__ == '__main__':
    main()
