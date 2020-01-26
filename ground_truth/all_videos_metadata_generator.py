"""Extract metadata information for all videos. 

"""

import os
import sys
import glob
import cv2
from resize import extract_frames, resize_video
from benchmarking.constants import RESOL_DICT
from ground_truth_generation_pipeline import gt_generation_pipeline
import shutil 
import json

RESOL_LIST = ['480p']
# RESOL_LIST = ['720p']
MODEL_LIST = ['Inception']
External_gt_source = '/mnt/data/zhujun/dataset/Inference_results/videos'

def extract_video_names(path):
    folder_list = [x for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]
    return folder_list

def get_video_resolution(vid):
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    return [int(width), int(height)]



def read_video_info(video_path):
    video_info = {}
    vid = cv2.VideoCapture(video_path)
    [width, height] = get_video_resolution(vid)
    fps = vid.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps
    minutes = int(duration/60)
    seconds = duration%60
    vid.release()

    video_info['duration'] = str(minutes) + ':' + str(seconds)
    video_info['resol'] = [width, height]
    video_info['fps'] = fps
    video_info['frame_count'] = frame_count
    return video_info

def video_existence_check(video, path):
    # check if the 720p version video exists
    # for cropped videos, resolution should be 360p
    flag = False
    video_info = {}
    mp4_names = glob.glob(path + '/*.mp4')
    if len(mp4_names) > 1:
        print('Datset {} has multipe videos.'.format(video))
    elif len(mp4_names) == 0:
        print('Datset {} has no video.'.format(video))
    else:
        video_info = read_video_info(mp4_names[0])
        width = video_info['resol'][0]
        height = video_info['resol'][1]
        if 'cropped' in video and width == 600 and height == 400:
            flag = True
        # Notice: resolution of dataset 'drift' is 1272x720
        elif 'drift' in video and width == 1272 and height == 720:
            flag = True
        else:
            if width == 1280 or height == 720:
                flag = True
            else:
                print('Dataset {} is not 720p'.format(video))

            if not os.path.exists(os.path.dirname(mp4_names[0]) + '/metadata.json'):
                print('{} not exists.'.format(os.path.dirname(mp4_names[0]) + '/metadata.json'))
    video_info['720p_video_exists'] = flag


    return video_info

def groundtruth_existence_check(video, path, video_info, fix):
    if 'cropped' in video:
        resolution_list = ['360p']
    else:
        resolution_list = RESOL_LIST
    
    gt_existence = {}
    for resol in resolution_list:
        gt_path = os.path.join(path, resol, 'profile')

        if not os.path.exists(gt_path):
            # check out external ground truth path
            another_gt_path = gt_path.replace(path, os.path.join(External_gt_source, video))            
            if not os.path.exists(another_gt_path):
                print('Dataset {} has no ground truth for resol {}'.format(video, resol))  
            else:
                print('Ground truth exists in another ground truth path:', video, resol)
            os.mkdir(gt_path)
        else:
            for model in MODEL_LIST:
                gt_existence[(resol, model)] = False
                gt_filename = os.path.join(gt_path, 'updated_gt_' + model + '_COCO_no_filter.csv')
                if not os.path.exists(gt_filename):
                    another_gt_path = gt_path.replace(path, os.path.join(External_gt_source, video))
                    another_gt_filename = os.path.join(another_gt_path, 'updated_gt_' + model + '_COCO_no_filter.csv')
                    if not os.path.exists(another_gt_filename):
                        print(video, resol, model)
                        #videoname, resol, model, gpu
                        if fix == True:
                            gt_generation_pipeline(video, resol, model, gpu='3')
                    else:
                        print('Copy gt for dataset {}, resol {}, model {}'.format(video, resol, model))
                        shutil.copyfile(another_gt_filename, gt_filename)
                        gt_existence[(resol, model)] = True
                else:
                    gt_existence[(resol, model)] = True
    
    video_info['ground_truth_existence'] = gt_existence #json.dumps(gt_existence)
    return video_info

def image_existence_check(video, path, video_info, fix):
    if 'cropped' in video:
        resolution_list = ['360p']  
    else:
        resolution_list = RESOL_LIST
    
    image_existence = {}
    for resol in resolution_list:
        image_path = os.path.join(path, resol)
        if not os.path.exists(image_path):
            print('Dataset {} does not have resolution {}'.format(video, resol))
            if fix:
                target_size = RESOL_DICT[resol]
                original_video_path = os.path.join(path, video + '.mp4')
                resized_video_path = os.path.join(image_path, video + '_' + resol + '.mp4')
                os.mkdir(image_path)
                resize_video(original_video_path, resized_video_path,
                    str(target_size[0])+':'+str(target_size[1]))
                extract_frames(resized_video_path, image_path)
                image_existence[resol] = True
            else:
                image_existence[resol] = False
        else:
            image_existence[resol] = True
    
    video_info['image_existence'] = image_existence # json.dumps(image_existence)

    return video_info


def main():
    DATA_path = '/mnt/data/zhujun/dataset/Youtube/'
    all_video_names = extract_video_names(DATA_path)

    all_video_checkresult = {}
    for video in all_video_names:
        video_path = os.path.join(DATA_path, video)
        video_info = video_existence_check(video, video_path)
        video_info = image_existence_check(video, video_path, video_info, fix=False)
        video_info = groundtruth_existence_check(video, video_path, video_info, fix=True)
        # print('{:20}: {}, {}'.format(video, video_info['720p_video_exists'],video_info['image_existence'] ))
        all_video_checkresult[video] = video_info
    # for key in all_video_checkresult:
    #     if all_video_checkresult[key]['frame_count'] > 22000:
    #         print(key, all_video_checkresult[key]['frame_count'])
    return

if __name__ == '__main__':
    main()