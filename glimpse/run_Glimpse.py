""" vigil scirpt  """
import argparse
import csv
import copy 
import pdb
import os
import sys
import glob
import numpy as np
from benchmarking.constants import OFFSET, Full_model, Original_resol, Glimpse_para1_dict, Glimpse_para2_list
from benchmarking.video import GeneralVideo
from benchmarking.glimpse.Glimpse_dev import Glimpse


def run_Glimpse_overfitting(info, video, para1_list, para2_list, segment_length, target_f1):
    path = info['path']
    name = video.video_name
    duration = video.duration
    result_path = os.path.join(path, 'overfitting', 'glimpse')
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    pipeline = Glimpse(para1_list, para2_list, os.path.join(result_path, 'glimpse_overfitting_log.csv'),
                       result_path, mode='frame select', target_f1=target_f1)  # mode='tracking'
    result_file = os.path.join(result_path, 'glimpse_overfitting_result.csv')
    with open(result_file, 'w', 1) as final_result_f:
        result_writer = csv.writer(final_result_f)
        header = ['video chunk', 'para1', 'para2', 'f1',
                  'frame rate', 'frame diff fps', 'tracking fps', 'bw']
        result_writer.writerow(header)

        # read ground truth and full model detection result
        # image name, detection result, ground truth

        if duration < segment_length:
            profile_length = int(duration // 3)
            start_frame = video.start_frame_index
            end_frame = video.end_frame_index
            profile_start_frame = start_frame
            profile_end_frame = end_frame
            test_start_frame =  start_frame
            test_end_frame = end_frame

            print("{} start={}, end={}".format(name, start_frame, end_frame))
            best_para1, best_para2 = pipeline.profile(name, video,
                                                    profile_start_frame,
                                                    profile_end_frame)

            print("best_para1={}, best_para2={}".format(best_para1,
                                                        best_para2))

            ideal_triggered_frame, f1, trigger_f1, pix_change_obj, \
                pix_change_bg, frame_diff_triggered, tracking_triggered, \
                frames_log = pipeline.evaluate(video, test_start_frame, test_end_frame,
                                            best_para1, best_para2)

            frames_triggered = frame_diff_triggered.union(tracking_triggered)
            bw = 0
            for frame_idx in frames_triggered:
                bw += video.get_frame_filesize(frame_idx)
            final_fps = len(frames_triggered) / (test_end_frame - test_start_frame + 1)
            frame_diff_fps = len(frame_diff_triggered) / \
                (test_end_frame - test_start_frame + 1)
            tracking_fps = len(tracking_triggered) / \
                (test_end_frame - test_start_frame + 1)

            result_writer.writerow(
                [name, best_para1, best_para2, f1, final_fps, frame_diff_fps,
                tracking_fps, bw])

            frames_log_file = os.path.join(
                result_path,
                name + '_{}_{}_frames_test_log.csv'.format(best_para1, best_para2))
            with open(frames_log_file, 'w') as f:
                frames_log_writer = csv.DictWriter(
                    f, ['frame id', 'frame diff', 'frame diff thresh',
                        'frame diff trigger', 'tracking error',
                        'tracking error thresh', 'tracking trigger',
                        'detection'])
                frames_log_writer.writeheader()
                frames_log_writer.writerows(frames_log)


        else:
            nb_short_videos = (video.frame_count - OFFSET *
                            video.frame_rate) // (segment_length *
                                                    video.frame_rate)

            for i in range(nb_short_videos):
                start_frame = video.start_frame_index + i*segment_length * \
                        video.frame_rate + OFFSET*video.frame_rate
                end_frame = video.start_frame_index + (i+1)*segment_length * \
                        video.frame_rate + OFFSET*video.frame_rate

                clip = name + '_' + str(i)

                profile_start_frame = start_frame
                profile_end_frame = end_frame # included in profile
                test_start_frame = start_frame
                test_end_frame = end_frame # included in test
                    

                print("{} start={}, end={}".format(clip, start, end))
                best_para1, best_para2 = pipeline.profile(clip, video,
                                                        profile_start_frame,
                                                        profile_end_frame)

                print("best_para1={}, best_para2={}".format(best_para1,
                                                            best_para2))

                ideal_triggered_frame, f1, trigger_f1, pix_change_obj, \
                    pix_change_bg, frame_diff_triggered, tracking_triggered, \
                    frames_log = pipeline.evaluate(video, test_start_frame, test_end_frame,
                                                best_para1, best_para2)
                # use the selected parameters for the next 5 mins
                frames_triggered = frame_diff_triggered.union(tracking_triggered)
                bw = 0
                for frame_idx in frames_triggered:
                    bw += video.get_frame_filesize(frame_idx)
                final_fps = len(frames_triggered) / (test_end_frame - test_start_frame + 1)
                frame_diff_fps = len(frame_diff_triggered) / \
                    (test_end_frame - test_start_frame + 1)
                tracking_fps = len(tracking_triggered) / \
                    (test_end_frame - test_start_frame + 1)
                result_writer.writerow(
                    [clip, best_para1, best_para2, f1, final_fps, frame_diff_fps,
                    tracking_fps, bw])

                frames_log_file = os.path.join(
                    result_path,
                    clip + '_{}_{}_frames_test_log.csv'.format(best_para1, best_para2))
                with open(frames_log_file, 'w') as f:
                    frames_log_writer = csv.DictWriter(
                        f, ['frame id', 'frame diff', 'frame diff thresh',
                            'frame diff trigger', 'tracking error',
                            'tracking error thresh', 'tracking trigger',
                            'detection'])
                    frames_log_writer.writeheader()
                    frames_log_writer.writerows(frames_log)

    return


def run_Glimpse_e2e(info, video, para1_list, para2_list, profile_length, segment_length, target_f1):

    path = info['path']
    name = video.video_name
    duration = video.duration
    result_path = os.path.join(path, 'e2e', 'glimpse')
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    pipeline = Glimpse(para1_list, para2_list, os.path.join(result_path, 'glimpse_e2e_log.csv'),
                       result_path, target_f1=target_f1)  # mode='tracking'
    result_file = os.path.join(result_path, 'glimpse_e2e_result.csv')
    with open(result_file, 'w', 1) as final_result_f:
        result_writer = csv.writer(final_result_f)
        header = ['video chunk', 'para1', 'para2', 'f1',
                  'frame rate', 'frame diff fps', 'tracking fps', 'bw']
        result_writer.writerow(header)

        # read ground truth and full model detection result
        # image name, detection result, ground truth

        if duration < segment_length:
            profile_length = int(duration // 3)
            start_frame = video.start_frame_index
            end_frame = video.end_frame_index
            profile_start_frame = start_frame
            profile_end_frame = profile_start_frame + video.frame_rate * \
                profile_length
            test_start_frame =  profile_end_frame + 1
            test_end_frame = end_frame

            print("{} start={}, end={}".format(name, start_frame, end_frame))
            best_para1, best_para2 = pipeline.profile(name, video,
                                                    profile_start_frame,
                                                    profile_end_frame)

            print("best_para1={}, best_para2={}".format(best_para1,
                                                        best_para2))

            ideal_triggered_frame, f1, trigger_f1, pix_change_obj, \
                pix_change_bg, frame_diff_triggered, tracking_triggered, \
                frames_log = pipeline.evaluate(video, test_start_frame, test_end_frame,
                                            best_para1, best_para2)

            frames_triggered = frame_diff_triggered.union(tracking_triggered)
            bw = 0
            for frame_idx in frames_triggered:
                bw += video.get_frame_filesize(frame_idx)
            final_fps = len(frames_triggered) / (test_end_frame - test_start_frame + 1)
            frame_diff_fps = len(frame_diff_triggered) / \
                (test_end_frame - test_start_frame + 1)
            tracking_fps = len(tracking_triggered) / \
                (test_end_frame - test_start_frame + 1)

            result_writer.writerow(
                [name, best_para1, best_para2, f1, final_fps, frame_diff_fps,
                tracking_fps, bw])

            frames_log_file = os.path.join(
                result_path,
                name + '_{}_{}_frames_test_log.csv'.format(best_para1, best_para2))
            with open(frames_log_file, 'w') as f:
                frames_log_writer = csv.DictWriter(
                    f, ['frame id', 'frame diff', 'frame diff thresh',
                        'frame diff trigger', 'tracking error',
                        'tracking error thresh', 'tracking trigger',
                        'detection'])
                frames_log_writer.writeheader()
                frames_log_writer.writerows(frames_log)


        else:
            nb_short_videos = (video.frame_count - OFFSET *
                            video.frame_rate) // (segment_length *
                                                    video.frame_rate)

            for i in range(nb_short_videos):
                start_frame = video.start_frame_index + i*segment_length * \
                        video.frame_rate + OFFSET*video.frame_rate
                end_frame = video.start_frame_index + (i+1)*segment_length * \
                        video.frame_rate + OFFSET*video.frame_rate

                clip = name + '_' + str(i)

                profile_start_frame = start_frame
                profile_end_frame = profile_start_frame + original_video.frame_rate * \
                    profile_length # included in profile
                test_start_frame = profile_end_frame + 1
                test_end_frame = end_frame # included in test
                    

                print("{} start={}, end={}".format(clip, start, end))
                best_para1, best_para2 = pipeline.profile(clip, video,
                                                        profile_start_frame,
                                                        profile_end_frame)

                print("best_para1={}, best_para2={}".format(best_para1,
                                                            best_para2))

                ideal_triggered_frame, f1, trigger_f1, pix_change_obj, \
                    pix_change_bg, frame_diff_triggered, tracking_triggered, \
                    frames_log = pipeline.evaluate(video, test_start_frame, test_end_frame,
                                                best_para1, best_para2)
                # use the selected parameters for the next 5 mins
                frames_triggered = frame_diff_triggered.union(tracking_triggered)
                bw = 0
                for frame_idx in frames_triggered:
                    bw += video.get_frame_filesize(frame_idx)
                final_fps = len(frames_triggered) / (test_end_frame - test_start_frame + 1)
                frame_diff_fps = len(frame_diff_triggered) / \
                    (test_end_frame - test_start_frame + 1)
                tracking_fps = len(tracking_triggered) / \
                    (test_end_frame - test_start_frame + 1)
                result_writer.writerow(
                    [clip, best_para1, best_para2, f1, final_fps, frame_diff_fps,
                    tracking_fps, bw])

                frames_log_file = os.path.join(
                    result_path,
                    clip + '_{}_{}_frames_test_log.csv'.format(best_para1, best_para2))
                with open(frames_log_file, 'w') as f:
                    frames_log_writer = csv.DictWriter(
                        f, ['frame id', 'frame diff', 'frame diff thresh',
                            'frame diff trigger', 'tracking error',
                            'tracking error thresh', 'tracking trigger',
                            'detection'])
                    frames_log_writer.writeheader()
                    frames_log_writer.writerows(frames_log)

    return

def run_Glimpse(info, profile_length, segment_length, target_f1):
    
    tstamp = 0
    path = info['path']

    original_video = GeneralVideo(info, Original_resol, Full_model, 
            filter_flag=True, merge_label_flag=True)
    name = original_video.video_name
    if 'waymo' in path:
        para1_list = Glimpse_para1_dict['waymo']
    else:
        para1_list = Glimpse_para1_dict[name]
    run_Glimpse_e2e(info, original_video, para1_list, Glimpse_para2_list, 
                    profile_length, segment_length, target_f1)
    run_Glimpse_overfitting(info, original_video, para1_list, Glimpse_para2_list, 
                    segment_length, target_f1)
    return


def main():
    dataset_info = {}
    dataset_info['path'] = '/mnt/data/zhujun/dataset/test/video'
    dataset_info['duration'] = int(9.9666666)
    dataset_info['resol'] = [1280, 720]
    dataset_info['frame_rate'] = 30
    dataset_info['frame_count'] = 299
    dataset_info['type'] = 'video'
    dataset_info['camera_type'] = 'static'

    profile_length = 10
    segment_length = 30
    run_Glimpse(dataset_info, 
                profile_length=profile_length, 
                segment_length=segment_length,
                target_f1=0.9)


if __name__ == '__main__':
    main()