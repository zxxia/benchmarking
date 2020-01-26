""" vigil scirpt  """
import argparse
import csv
import copy 
import pdb
import os
import sys
import glob
from benchmarking.constants import OFFSET, Full_model, Original_resol
from benchmarking.video import GeneralVideo
from benchmarking.vigil.Vigil import Vigil, mask_video_ffmpeg
from benchmarking.ground_truth.ground_truth_generation_pipeline import gt_generation_pipeline

# DT_ROOT = '/mnt/data/zhujun/dataset/Youtube'
# New_DT_ROOT = '/mnt/data/zhujun/dataset/Inference_results/videos'
# Vigil_DATA_ROOT = '/mnt/data/zhujun/dataset/Vigil_result/'


# SMALL_MODEL_PATH = '/mnt/data/zhujun/dataset/NoScope_finetuned_models'



def run_Vigil_mask(info, gpu_num, local_model):
    path = info['path']
    resol = '720p'
    output_path = os.path.join(path, 'masked_images', resol)
    video = GeneralVideo(info, resol, local_model, filter_flag=True, merge_label_flag=False)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print('start masking out background of ', path)
        mask_video_ffmpeg(video, 0.1, 0.1, save_path=output_path)
        print('finish masking out background of', path)
    
    if info['type'] == 'png':
        extension = 'png'
    else:
        extension = 'jpg'
    masked_gt = os.path.join(output_path, 'profile')
    model = 'FasterRCNN'
    if not os.path.exists(masked_gt):
        gt_generation_pipeline(output_path, resol, model, extension, gpu_num)
        for record_file in glob.glob(masked_gt + '/*.record'):
            os.remove(record_file)
    print('finish running FasterRCNN on masked images.')
    return os.path.join(path, 'masked_images')

def run_Vigil_e2e(original_video, masked_video, profile_length, segment_length, e2e_result_path):

    name = original_video.video_name
    profile_video_path = e2e_result_path.replace('vigil', 'profile')
    test_video_path = e2e_result_path.replace('vigil', 'test')
    masked_Video_path = e2e_result_path
    pipeline = Vigil()
    if not os.path.exists(profile_video_path):
        os.makedirs(profile_video_path)
    if not os.path.exists(test_video_path):
        os.makedirs(test_video_path)
    if not os.path.exists(e2e_result_path):
        os.makedirs(e2e_result_path)

    with open(e2e_result_path + '/vigil_e2e_result_' + name + '.csv', 'w', 1) as f_out:
        writer = csv.writer(f_out)
        writer.writerow(['video', 'frame_bw', 'f1', 'video_bw', 'original_bw'])
        if original_video.duration < segment_length:
            profile_length = int(original_video.duration // 3)
            start_frame = original_video.start_frame_index
            end_frame = original_video.end_frame_index
            profile_start_frame = start_frame
            profile_end_frame = profile_start_frame + original_video.frame_rate * \
                profile_length
            test_start_frame =  profile_end_frame + 1
            test_end_frame = end_frame
            bw, f1, video_bw, original_bw = pipeline.evaluate(name, masked_video, original_video,
                                    [test_start_frame, test_end_frame],
                                    test_video_path,
                                    e2e_result_path)

            print('{}, start={}, end={}, f1={}, bw={}, video_bw={}'
                .format(name, start_frame, end_frame, f1, bw, video_bw))
            writer.writerow([name, bw, f1, video_bw, original_bw])

        else:
        
            nb_short_videos = original_video.frame_count//(segment_length *
                                                    original_video.frame_rate)
            for i in range(nb_short_videos):
                clip = name + '_' + str(i)
                start_frame = original_video.start_frame_index + i*segment_length * \
                    original_video.frame_rate + OFFSET*original_video.frame_rate
                end_frame = original_video.start_frame_index + (i+1)*segment_length * \
                    original_video.frame_rate + OFFSET*original_video.frame_rate
                profile_start_frame = start_frame
                profile_end_frame = profile_start_frame + original_video.frame_rate * \
                    profile_length # included in profile
                test_start_frame = profile_end_frame + 1
                test_end_frame = end_frame # included in test
                

                bw, f1, video_bw, original_bw = pipeline.evaluate(clip, masked_video, original_video,
                                        [test_start_frame, test_end_frame],
                                        test_video_path,
                                        e2e_result_path)

                print('{}, start={}, end={}, f1={}, bw={}, video_bw={}'
                    .format(clip, start_frame, end_frame, f1, bw, video_bw))
                writer.writerow([clip, bw, f1, video_bw, original_bw])

    return

def run_Vigil_overfitting(original_video, masked_video, segment_length, overfitting_result_path):

    name = original_video.video_name
    profile_video_path = overfitting_result_path.replace('vigil', 'profile')
    masked_Video_path = overfitting_result_path
    pipeline = Vigil()
    if not os.path.exists(profile_video_path):
        os.makedirs(profile_video_path)
    if not os.path.exists(masked_Video_path):
        os.makedirs(masked_Video_path)

    with open(overfitting_result_path + '/vigil_overfitting_result_' + name + '.csv', 'w', 1) as f_out:
        writer = csv.writer(f_out)
        writer.writerow(['video', 'frame_bw', 'f1', 'video_bw', 'original_bw'])
        if original_video.duration < segment_length:
            start_frame = original_video.start_frame_index
            end_frame = original_video.end_frame_index
            profile_start_frame = start_frame
            profile_end_frame = end_frame
            test_start_frame =  start_frame
            test_end_frame = end_frame
            bw, f1, video_bw, original_bw = pipeline.evaluate(name, masked_video, original_video,
                                    [test_start_frame, test_end_frame],
                                    profile_video_path,
                                    masked_Video_path)

            print('{}, start={}, end={}, f1={}, bw={}, video_bw={}'
                .format(name, start_frame, end_frame, f1, bw, video_bw))
            writer.writerow([name, bw, f1, video_bw, original_bw])

        else:
        
            nb_short_videos = original_video.frame_count//(segment_length *
                                                    original_video.frame_rate)
            for i in range(nb_short_videos):
                clip = name + '_' + str(i)
                start_frame = original_video.start_frame_index + i*segment_length * \
                    original_video.frame_rate + OFFSET*original_video.frame_rate
                end_frame = original_video.start_frame_index + (i+1)*segment_length * \
                    original_video.frame_rate + OFFSET*original_video.frame_rate
                profile_start_frame = start_frame
                profile_end_frame =  end_frame# included in profile
                test_start_frame =  start_frame
                test_end_frame = end_frame # included in test
                

                bw, f1, video_bw, original_bw = pipeline.evaluate(clip, masked_video, original_video,
                                        [test_start_frame, test_end_frame],
                                        profile_video_path,
                                        masked_Video_path)

                print('{}, start={}, end={}, f1={}, bw={}, video_bw={}'
                    .format(clip, start_frame, end_frame, f1, bw, video_bw))
                writer.writerow([clip, bw, f1, video_bw, original_bw])

    return

def run_Vigil(info, gpu_num, local_model, profile_length=10, segment_length=30):

    masked_image_path = run_Vigil_mask(info, gpu_num, local_model)

    pipeline = Vigil()
    path = info['path']
    original_video = GeneralVideo(info, Original_resol, Full_model, 
                    filter_flag=True, merge_label_flag=True)
    masked_info = copy.deepcopy(info)
    masked_info['path'] = masked_image_path
    masked_video = GeneralVideo(masked_info, Original_resol, Full_model, 
                    filter_flag=True, merge_label_flag=True)
    print('Starting running vigil e2e results.')
    e2e_result_path = os.path.join(path, 'e2e', 'vigil')
    run_Vigil_e2e(original_video, masked_video, profile_length, segment_length, e2e_result_path)
    print('Starting running vigil overfitting results.')
    overfitting_result_path = os.path.join(path, 'overfitting', 'vigil')
    run_Vigil_overfitting(original_video, masked_video, segment_length, overfitting_result_path)
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
    duration = dataset_info['duration']

    profile_length = 10
    segment_length = 30
    run_Vigil(dataset_info, '1', local_model='Inception', 
                profile_length=profile_length, 
                segment_length=segment_length)


if __name__ == '__main__':
    main()
