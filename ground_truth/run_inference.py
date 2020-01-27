"""Extract metadata information for all videos. 

"""

import os
import sys
import glob
import cv2
from benchmarking.ground_truth.resize import extract_frames, resize_video, resize_image
from benchmarking.constants import RESOL_DICT, RESOL_LIST, MODEL_LIST
from benchmarking.ground_truth.ground_truth_generation_pipeline import gt_generation_pipeline
import shutil 
import json





def get_inference_results(info, gpu_num):
    path = info['path']
    if info['type'] == 'png':
        extension = 'png'
    else:
        extension = 'jpg'

    for model in ['mobilenet', 'Inception', 'FasterRCNN50']:#MODEL_LIST:
        resol = '720p'
        gt_path = os.path.join(path, resol, 'profile')
        if not os.path.exists(gt_path):
            os.mkdir(gt_path)
        gt_filename = os.path.join(gt_path, 'updated_gt_' + model + '_COCO_no_filter.csv')        
        if not os.path.exists(gt_filename):
            gt_generation_pipeline(os.path.join(path, resol), resol, model, extension, gpu_num)
        if os.path.exists(gt_path + '/input.record'):
            os.remove(gt_path + '/input.record')

    # tmp = RESOL_LIST.copy()
    # tmp.remove('720p')
    # for resol in tmp:
    #     gt_path = os.path.join(path, resol, 'profile')
    #     if not os.path.exists(gt_path):
    #         os.mkdir(gt_path)
    #     model = 'FasterRCNN'
    #     gt_filename = os.path.join(gt_path, 'updated_gt_' + model + '_COCO_no_filter.csv')
    #     if not os.path.exists(gt_filename):
    #         gt_generation_pipeline(os.path.join(path, resol), resol, model, extension, gpu_num)   
    #     if os.path.exists(gt_path + '/input.record'):
    #         os.remove(gt_path + '/input.record')
    
    return



        

def prepare_images(info):
    path = info['path']
    mp4_file = os.path.join(path, info['video_name'] + '.mp4')
    
    for resol in RESOL_LIST:
        image_path = os.path.join(path, resol)
        if not os.path.exists(image_path):
            print('Dataset {} does not have resolution {}'.format(info['video_name'], resol))
            target_size = RESOL_DICT[resol]
            original_video_path = mp4_file
            resized_video_path = os.path.join(image_path, info['video_name'] + '_' + resol + '.mp4')
            os.mkdir(image_path)
            resize_video(original_video_path, resized_video_path,
                str(target_size[0])+':'+str(target_size[1]))
            extract_frames(resized_video_path, image_path)
            os.remove(resized_video_path)
    return 



def run_video_inference(info, gpu_num):
    prepare_images(info)
    get_inference_results(info, gpu_num)
    return


def prepare_images_from_image(info):
    path = info['path']
    for resol in RESOL_LIST:
        image_path = os.path.join(path, resol)
        if not os.path.exists(image_path):
            target_size = RESOL_DICT[resol]
            os.mkdir(image_path)
            for imagename in glob.glob(path + '/*' + info['type']):
                resized_imagename = os.path.join(image_path, os.path.basename(imagename))
                resize_image(imagename, resized_imagename,
                str(target_size[0])+':'+str(target_size[1]))
    return

def run_image_inference(info, gpu_num):
    prepare_images_from_image(info)
    get_inference_results(info, gpu_num)
    return


def run_inference(info, gpu_num):
    if info['type'] == 'video':
        run_video_inference(info, gpu_num)
    else:
        run_image_inference(info, gpu_num)
    return

def main():
    dataset_info = []
    dataset_info['path'] = '/mnt/data/zhujun/dataset/test/video/'    
    dataset_info['duration'] = "0:9.966666666666667"
    dataset_info['resol'] = [1280, 720]
    dataset_info['frame_rate'] = 30
    dataset_info['frame_count'] = 299
    dataset_info['type'] = 'video'
    dataset_info['camera_type'] = 'static'
    
    run_inference(dataset_info)
    return

if __name__ == '__main__':
    main()