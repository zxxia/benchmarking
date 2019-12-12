"""Glimpse with frame selectction.

Frame difference trigger is implemented. But tracking is not implemented.
"""
import argparse
import numpy as np
from glimpse import pipeline_frame_select, \
    filter_video_detections, compute_target_frame_rate
# object_appearance, pipeline_perfect_tracking
from utils.model_utils import load_full_model_detection
from utils.utils import load_metadata
from constants import CAMERA_TYPES


PARA1_LIST_DICT = {
    'crossroad': np.arange(30, 52, 2),
    'crossroad2': np.arange(30, 52, 2),
    'crossroad3': np.arange(80, 120, 3),
    'crossroad4': np.arange(40, 82, 2),
    'drift': np.arange(290, 400, 10),
    'driving1': np.arange(15, 45, 2),
    'driving2': np.arange(10, 60, 2),
    'driving_downtown': np.arange(4, 20, 2),
    'highway': np.arange(35, 60, 1),
    'highway_normal_traffic': np.arange(34, 40, 2),
    'jp': np.arange(30, 40, 2),
    'jp_hw': np.arange(30, 40, 2),
    'lane_split': np.arange(6, 14, 2),
    'motorway': np.arange(2, 6, 0.5),
    'nyc': np.arange(2, 60, 2),
    'park': np.arange(2, 40, 2),
    'russia': np.arange(100, 400, 10),
    'russia1': np.arange(100, 400, 10),
    'traffic': np.arange(6, 15, 1),
    'tw': np.arange(25, 80, 2),
    'tw1': np.arange(25, 80, 2),
    # 'tw_road': np.arange(15, 45, 5),
    'tw_under_bridge': np.arange(350, 450, 10),
}
PARA2_LIST = [0.5]


def parse_args():
    """Parse the input arguments required by the code."""
    parser = argparse.ArgumentParser(
        description="Glimpse with perfect tracking")
    parser.add_argument("--path", type=str, required=True,
                        help="path contains all datasets")
    parser.add_argument("--video", type=str, required=True,
                        help="video name")
    parser.add_argument("--metadata", type=str,  # required=True,
                        help="metadata file in Json")
    parser.add_argument("--output", type=str, required=True,
                        help="output result file")
    parser.add_argument("--log", type=str,
                        help="profiling log file")
    parser.add_argument("--short_video_length", type=int, required=True,
                        help="short video length in seconds")
    parser.add_argument("--profile_length", type=int, required=True,
                        help="profile length in seconds")
    parser.add_argument("--offset", type=int, default=0,
                        help="offset from beginning of the video in seconds")
    parser.add_argument("--target_f1", type=float, help="target F1 score")
    # parser.add_argument("--fps", type=int, default=0, help="frame rate")
    # parser.add_argument("--resolution", nargs='+', type=int,
    #                     default=[], action='store', help="video resolution")
    parser.add_argument("--format", type=str, default='{:06d}.jpg',
                        help="image name format")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    path = args.path
    video_name = args.video
    output_file = args.output
    log_file = args.log
    metadata_file = args.metadata
    short_video_length = args.short_video_length
    profile_length = args.profile_length
    offset = args.offset
    target_f1 = args.target_f1
    img_name_format = args.format

    f_trace = open('{}_trace.csv'.format(video_name), 'w')
    f_trace.write('frame id,timestamp,trigger\n')
    tstamp = 0

    # choose the first 3 mins to get the best frame diff thresh
    with open(output_file, 'w', 1) as final_result_f:
        header = 'video chunk,para1,para2,f1,frame rate,' \
            'ideal frame rate,trigger f1\n'
        final_result_f.write(header)
        # for video_type in DATASET_LIST:
        f_profile = open(log_file, 'w', 1)
        f_profile.write(header)
        # f_trigger_log = open(
        # 'profile_perfect_tracking_waymo/trigger_log_{}.json'
        # .format(video_name), 'w')

        metadata = load_metadata(metadata_file)
        # resolution = metadata['resolution']
        fps = metadata['frame rate']
        # frame_count = metadata['frame count']

        # read ground truth and full model detection result
        # image name, detection result, ground truth

        # img_path = path + str(resolution[1]) + 'p/'
        img_path = path + '720p/'
        annot_file = '/data/zxxia/benchmarking/results/videos/{}/720p/profile/updated_gt_FasterRCNN_COCO_no_filter.csv'.format(
            video_name)
        gt_annot, frame_end = load_full_model_detection(annot_file)
        if video_name in CAMERA_TYPES['static']:
            gt_annot = filter_video_detections(gt_annot,
                                               width_range=(0, 1280/2),
                                               height_range=(0, 720/2))

        nb_short_videos = (frame_end-offset*fps)//(short_video_length*fps)

        if nb_short_videos == 0:
            nb_short_videos = 1

        for i in range(nb_short_videos):
            start = i*short_video_length*fps+1+offset*fps
            end = (i+1)*short_video_length*fps+offset*fps

            if end > frame_end:
                end = frame_end
                clip = video_name
            else:
                clip = video_name + '_' + str(i)

            test_f1_list = []
            test_fps_list = []

            profile_start = start
            profile_end = min(start+profile_length*fps-1, end)
            test_start = profile_end + 1
            test_end = end
            print("{} {} start={}, end={}".format(clip, img_path, start, end))

            # Run inference on the first 30s video
            f1_diff = 1
            min_f1_gt_target = 1.0
            # the minimum f1 score which is greater than
            # or equal to target f1(e.g. 0.9)
            min_fps = fps

            best_para1 = -1
            best_para2 = -1
            best_trigger_f1 = 0
            best_pix_change_obj = 0
            best_pix_change_bg = 0
            best_frames_triggered = None
            # best_video_trigger_log = []

            # for para1 in [para1_dict[clip]]:
            for para1 in PARA1_LIST_DICT[video_name]:
                para2 = PARA2_LIST[0]
                # for para2 in para2_list:
                # larger para1, smaller thresh, easier to be triggered
                frame_difference_thresh = 1280*720/para1
                tracking_error_thresh = para2
                # images start from index 1
                triggered_frame, ideal_triggered_frame, f1, \
                    trigger_f1, video_trigger_log, pix_change_obj, \
                    pix_change_bg, frames_triggered = \
                    pipeline_frame_select(img_path, gt_annot,
                                          profile_start, profile_end,
                                          frame_difference_thresh,
                                          tracking_error_thresh,
                                          img_name_format, view=False,
                                          mask_flag=True)
                current_fps = triggered_frame/profile_length
                ideal_fps = ideal_triggered_frame/profile_length
                print('para1 = {}, para2 = {}, '
                      'Profiled f1 = {}, Profiled perf = {}, Ideal perf={}'
                      .format(para1, para2, f1, current_fps/fps,
                              ideal_fps/fps))
                f_profile.write(','.join([clip, str(para1), str(para2),
                                          str(f1), str(current_fps/fps),
                                          str(ideal_fps/fps),
                                          str(trigger_f1),
                                          str(pix_change_obj),
                                          str(pix_change_bg)])+'\n')
                test_f1_list.append(f1)
                test_fps_list.append(current_fps)

                if abs(f1 - target_f1) < f1_diff:
                    f1_diff = abs(f1-target_f1)
                    min_f1_gt_target = f1
                    min_fps = current_fps
                    # record the best config
                    best_para1 = para1
                    best_para2 = para2
                    best_trigger_f1 = trigger_f1
                    best_pix_change_obj = pix_change_obj
                    best_pix_change_bg = pix_change_bg
                    best_frames_triggered = frames_triggered

            test_f1_list.append(1.0)
            test_fps_list.append(fps)
            final_fps, f1_left, f1_right, fps_left, fps_right =\
                compute_target_frame_rate(test_fps_list, test_f1_list)

            print("best_para1 = {}, best_para2={}"
                  .format(best_para1, best_para2))

            frame_difference_thresh = 1280*720/best_para1
            tracking_error_thresh = best_para2
            triggered_frame, ideal_triggered_frame, f1, \
                trigger_f1, video_trigger_log, pix_change_obj, \
                pix_change_bg, frames_triggered = \
                pipeline_frame_select(img_path, gt_annot,
                                      test_start, test_end,
                                      frame_difference_thresh,
                                      tracking_error_thresh,
                                      img_name_format, view=False,
                                      mask_flag=True)
            # use the selected parameters for the next 5 mins
            final_result_f.write(','.join([clip,
                                           str(best_para1),
                                           str(best_para2),
                                           str(min_f1_gt_target),
                                           str(final_fps/fps),
                                           str(ideal_fps/fps),
                                           str(best_trigger_f1),
                                           str(best_pix_change_obj),
                                           str(best_pix_change_bg),
                                           str(f1_left), str(f1_right),
                                           str(fps_left),
                                           str(fps_right)]) + '\n')
            for idx in range(start, end + 1):
                if idx in set(range(profile_start, profile_end+1)).union(frames_triggered):
                    f_trace.write(
                        ','.join([str(idx), str(tstamp), str(1)])+'\n')
                else:
                    f_trace.write(
                        ','.join([str(idx), str(tstamp), str(0)]) + '\n')
                tstamp += 1/fps
            # final_result_f.write(','.join([clip,
            #                                str(best_para1),
            #                                str(best_para2),
            #                                str(min_f1_gt_target),
            #                                str(min_fps/fps),
            #                                str(ideal_fps/fps),
            #                                str(best_trigger_f1),
            #                                str(best_pix_change_obj),
            #                                str(best_pix_change_bg)])+'\n')
            # json.dump(best_video_trigger_log, f_trigger_log,
            #           sort_keys=True, indent=4)
        f_profile.close()
        f_trace.close()

        # f_trigger_log.close()

# This is the old parameters for perfect tracking without masking
# PARA1_LIST_DICT = {
#         'crossroad': np.arange(15, 40, 5),
#         'crossroad2': np.arange(15, 40, 5),
#         'crossroad3': np.arange(20, 50, 5),
#         'crossroad4': np.arange(20, 45, 5),
#         'drift': np.arange(20, 40, 5),
#         'driving1': np.arange(10, 30, 2),
#         'driving2': np.arange(2, 10, 1),
#         'driving_downtown': np.arange(5, 15, 3),
#         'highway': np.arange(20, 40, 5),
#         'highway_normal_traffic': np.arange(15, 40, 5),
#         'jp': [25],
#         'jp_hw': [25],
#         'lane_split': np.arange(4, 10, 2),
#         'motorway': np.arange(2, 4, 0.5),
#         'nyc': np.arange(8, 10, 1),
#         'park': np.arange(2, 5, 1),
#         'russia': np.arange(15, 40, 5),
#         'russia1': np.arange(15, 60, 5),
#         'traffic': np.arange(6, 8, 1),
#         'tw': np.arange(15, 40, 5),
#         'tw1': np.arange(15, 40, 5),
#         'tw_road': np.arange(15, 40, 5),
#         'tw_under_bridge': np.arange(90, 120, 10),}


# This is the parameter set which only new viechle pixel change is considered
# PARA1_LIST_DICT = {
#         'crossroad': np.arange(35, 70, 5),
#         'crossroad2': np.arange(80, 90, 2),
#         'crossroad3': np.arange(45, 70, 5),
#         'crossroad4': np.arange(60, 85, 5),
#         'drift': np.arange(145, 160, 5),
#         'driving1': np.arange(10, 30, 2),
#         'driving2': np.arange(2, 10, 1),
#         'driving_downtown': np.arange(70, 100, 10),
#         'highway': np.arange(75, 82, 2),
#         'highway_normal_traffic': np.arange(15, 40, 5),
#         'jp': [27, 30],
#         'jp_hw': [25],
#         'lane_split': np.arange(6, 10, 2),
#         'motorway': np.arange(18, 22, 1),
#         'nyc': np.arange(6, 10, 1),
#         'park': np.arange(2, 8, 1),
#         'russia': np.arange(70, 90, 2),
#         'russia1': np.arange(65, 80, 5),
#         'traffic': np.arange(10, 22, 2),
#         'tw': np.arange(15, 50, 5),
#         'tw1': np.arange(15, 65, 10),
#         # 'tw_road': np.arange(15, 45, 5),
#         'tw_under_bridge': np.arange(200, 250, 10),
#         }
if __name__ == '__main__':
    main()
