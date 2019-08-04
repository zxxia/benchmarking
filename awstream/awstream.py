from collections import defaultdict
import subprocess
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import sys
import json
from videostorm.VideoStorm_temporal import load_full_model_detection, eval_single_image
from my_utils import interpolation, compute_f1, load_metadata
import cv2
import os
import pdb

# PATH = '/mnt/data/zhujun/new_video/'
PATH = '/mnt/data/zhujun/dataset/Youtube/'
TEMPORAL_SAMPLING_LIST = [20,15,10,5,4,3,2.5,2,1.8,1.5,1.2,1]
DATASET_LIST = sorted(['jp', 'highway', 'motorway'])#sorted(['traffic', 'jp_hw', 'russia', 'tw_road', 
               #        'tw_under_bridge', 'highway_normal_traffic', 'nyc', 
               #        'lane_split', 'tw', 'tw1', 'jp', 'russia1','drift', 
               #        'park', 'walking', 'highway', 'crossroad2', 'crossroad', 
               #        'crossroad3', 'crossroad4', 'driving1', 'driving2',
               #        'driving_downtown', 'block', 'block1', 'motorway'])


IMAGE_RESOLUTION_DICT = {'360p': [480, 360],
                         '480p': [640, 480],
                         '540p': [960, 540],
                         '576p': [1024, 576],
                        }

SHORT_VIDEO_LENGTH = 2*60 #seconds
IOU_THRESH = 0.5
TARGET_F1 = 0.9
PROFILE_LENGTH = 30 #seconds

RESOLUTION_LIST = ['original', '540p', '480p', '360p']


def compress_images_to_video(list_file: str, frame_rate: str, resolution: str, 
                             quality: int, output_name: str):
    '''
    Compress a set of frames to a video.
    '''
    cmd = ['ffmpeg', '-y', '-r', str(frame_rate), '-f', 'concat', '-safe', '0',
           '-i', list_file, '-s', str(resolution), '-vcodec', 'libx264', 
           '-crf', str(quality), '-pix_fmt', 'yuv420p', '-hide_banner', 
           output_name]
    subprocess.run(cmd)

def compute_video_size(dataset, start, end, 
    target_frame_rate, frame_rate, resolution):
    metadata = load_metadata(PATH + dataset + '/metadata.json')

    if resolution == 'original':
        img_path = PATH + dataset
        image_resolution = metadata['resolution']
    else:
        img_path = PATH + dataset + '/' + resolution + '/'
        image_resolution = IMAGE_RESOLUTION_DICT[resolution]

    sample_rate = frame_rate/target_frame_rate

    # Create a tmp list file contains all the selected iamges
    tmp_list_file = 'list.txt'
    with open(tmp_list_file, 'w') as f:
        for img_index in range(start, end + 1):
            # based on sample rate, decide whether this frame is sampled
            if img_index%sample_rate >= 1:
                continue
            else:
                line = 'file \'{}/{:06}.jpg\'\n'.format(img_path, img_index)
                f.write(line)


    frame_size = str(image_resolution[0]) + 'x' + str(image_resolution[1])
    compress_images_to_video(tmp_list_file, target_frame_rate, frame_size, 
                             25, 'tmp.mp4')
    video_size = os.path.getsize("tmp.mp4")
    os.remove('tmp.mp4')
    os.remove(tmp_list_file)
    print('target frame rate={}, target image resolution={}. video size={}'.format(target_frame_rate, image_resolution, video_size))
    return video_size


def profile(dataset, frame_rate, start_frame, end_frame, f_profile):
    result = {}
    metadata = load_metadata(PATH + dataset + '/metadata.json')
    # choose resolution
    # choose frame rate
    
    for resolution in RESOLUTION_LIST:
        F1_score_list = []
        if resolution == 'original':
            dt_file = PATH + dataset + '/profile/updated_gt_FasterRCNN_COCO.csv' 
            image_resolution = metadata['resolution']
            gt_file = PATH + dataset + '/profile/updated_gt_FasterRCNN_COCO.csv'    
            gt, num_of_frames = load_full_model_detection(gt_file)

        else:
            dt_file = PATH + dataset + '/' + resolution + '/profile/updated_gt_FasterRCNN_COCO.csv'
            image_resolution = IMAGE_RESOLUTION_DICT[resolution]
            gt_file = dt_file.replace('updated_gt_FasterRCNN_COCO.csv', 'gt_' + resolution + '.csv')
            height = image_resolution[1]
            gt, _ = load_full_model_detection(gt_file)


        height = image_resolution[1]
        full_model_dt, num_of_frames = load_full_model_detection(dt_file)
        
        for sample_rate in TEMPORAL_SAMPLING_LIST:
            img_cn = 0
            tp = defaultdict(int)
            fp = defaultdict(int)
            fn = defaultdict(int)
            save_dt = []
            for img_index in range(start_frame, end_frame+1):
                dt_boxes_final = []
                current_full_model_dt = full_model_dt[img_index]
                current_gt = gt[img_index]
                # based on sample rate, decide whether this frame is sampled
                if img_index%sample_rate >= 1:
                    # this frame is not sampled, so reuse the last saved
                    # detection result
                    dt_boxes_final = [box for box in save_dt]
                else:
                    # this frame is sampled, so use the full model result
                    dt_boxes_final = [box for box in current_full_model_dt]
                    save_dt = [box for box in dt_boxes_final]

                tp[img_index], fp[img_index], fn[img_index] = \
                        eval_single_image(current_gt, dt_boxes_final)

                # print(img_index, tp[img_index], fp[img_index],fn[img_index])
            tp_total = sum(tp.values())
            fp_total = sum(fp.values())
            fn_total = sum(fn.values())

            f1 = compute_f1(tp_total, fp_total, fn_total)
            bw = image_resolution[0] * image_resolution[1]
            #compute_video_size(dataset, start_frame, 
                 #                   start_frame+chunk_length*frame_rate, 
                 #                   frame_rate, frame_rate, resolution)

            f_profile.write(','.join([resolution, str(sample_rate), str(f1), 
                                      '\n'])) 
            print('resolution={} and sample rate={} ==> profiled f1={}'.format(resolution, sample_rate, f1))
            F1_score_list.append(f1)

        frame_rate_list = [frame_rate/x for x in TEMPORAL_SAMPLING_LIST]
        current_f1_list = F1_score_list

        if current_f1_list[-1] < TARGET_F1:
            target_frame_rate = None
        else:
            index = next(x[0] for x in enumerate(current_f1_list) if x[1] > TARGET_F1)
            if index == 0:
                target_frame_rate = frame_rate_list[0]
            else:
                point_a = (current_f1_list[index-1], frame_rate_list[index-1])
                point_b = (current_f1_list[index], frame_rate_list[index])
                target_frame_rate  = interpolation(point_a, point_b, TARGET_F1)
        print("Resolution = {} and target frame rate = {}".format(resolution, target_frame_rate))
        result[resolution] = target_frame_rate
    return result


def select_best_config(dataset, frame_rate, start_frame, end_frame, configs):

    # select best profile
    origin_bw = compute_video_size(dataset, start_frame, end_frame, frame_rate, 
                                   frame_rate, 'original')

    min_bw = origin_bw #metadata['resolution'][0]*metadata['resolution'][1]*standard_frame_rate
    best_resol = 'original'
    best_frame_rate = frame_rate
    for resolution in configs.keys():
        target_frame_rate = configs[resolution]

        if target_frame_rate == None:
            continue
        video_size = compute_video_size(dataset, start_frame, end_frame, 
                                        target_frame_rate, frame_rate, 
                                        resolution)
        bw = video_size
 
        if bw < min_bw:
            best_resol = resolution
            best_frame_rate = target_frame_rate
            min_bw = bw
     
    return best_resol, best_frame_rate, min_bw/origin_bw


def profile_eval(dataset, frame_rate, best_resolution, best_sample_rate,
                 start_frame, end_frame):
    metadata = load_metadata(PATH + dataset + '/metadata.json')

    if best_resolution == 'original':
        dt_file = PATH + dataset + '/profile/updated_gt_FasterRCNN_COCO.csv' 
        image_resolution = metadata['resolution']
        gt_file = PATH + dataset + '/profile/updated_gt_FasterRCNN_COCO.csv'    
        gt, num_of_frames = load_full_model_detection(gt_file)

    else:
        dt_file = PATH + dataset + '/' + best_resolution + '/profile/updated_gt_FasterRCNN_COCO.csv'
        image_resolution = IMAGE_RESOLUTION_DICT[best_resolution]
        gt_file = dt_file.replace('updated_gt_FasterRCNN_COCO.csv', 'gt_' + best_resolution + '.csv')
        height = image_resolution[1]
        gt, _ = load_full_model_detection(gt_file)

    height = image_resolution[1]

    full_model_dt, _ = load_full_model_detection(dt_file)

    img_cn = 0
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    save_dt = []

    for img_index in range(start_frame, end_frame):
        dt_boxes_final = []
        current_full_model_dt = full_model_dt[img_index]
        current_gt = gt[img_index]

        # based on sample rate, decide whether this frame is sampled
        if img_index%best_sample_rate >= 1:
            # this frame is not sampled, so reuse the last saved
            # detection result
            dt_boxes_final = [box for box in save_dt]

        else:
            # this frame is sampled, so use the full model result
            dt_boxes_final = [box for box in current_full_model_dt]
            save_dt = [box for box in dt_boxes_final]

        tp[img_index], fp[img_index], fn[img_index] = eval_single_image(current_gt, dt_boxes_final)   
                            
    tp_total = sum(tp.values())
    fp_total = sum(fp.values())
    fn_total = sum(fn.values())

    return compute_f1(tp_total, fp_total, fn_total)

def main():

    with open('awstream_motivation_result_updated.csv', 'w') as f:
        f.write('dataset,best_resolution,f1,frame_rate,bandwidth\n')
        for dataset in DATASET_LIST:
            f_profile = open('awstream_motivation_profile_{}.csv'.format(dataset), 'w')
            metadata = load_metadata(PATH + dataset + '/metadata.json')
            height = metadata['resolution'][1]
            # load detection results of fasterRCNN + full resolution + 
            #highest frame rate as ground truth
            frame_rate = metadata['frame rate']
            num_of_short_videos = metadata['frame count']//(SHORT_VIDEO_LENGTH*frame_rate)
            print('Processing', dataset)
            test_bw_list = list()
            test_f1_list = list()
            for i in range(num_of_short_videos):
                start_frame = i * (SHORT_VIDEO_LENGTH*frame_rate) + 1
                end_frame = (i+1) * (SHORT_VIDEO_LENGTH*frame_rate)

                # use 30 seconds video for profiling
                #pdb.set_trace()
                profile_start_frame = start_frame
                profile_end_frame = start_frame + frame_rate * PROFILE_LENGTH - 1
                configs = profile(dataset, frame_rate, profile_start_frame, 
                                  profile_end_frame, f_profile)

                best_resol, best_fr, best_bw = select_best_config(dataset, 
                                                                  frame_rate, 
                                                                  profile_start_frame,
                                                                  profile_end_frame, 
                                                                  configs)
                
                print("Finished profiling on frame [{},{}].".format(profile_start_frame, 
                                                                profile_end_frame))

                # test on the whole video
                best_sample_rate = frame_rate / best_fr
                #pdb.set_trace()
                test_start_frame = profile_end_frame + 1
                test_end_frame = end_frame
                f1 = profile_eval(dataset, frame_rate, best_resol, 
                                  best_sample_rate, test_start_frame, 
                                  test_end_frame)

                print("Finished testing on frame [{},{}].".format(test_start_frame, test_end_frame))
                test_bw_list.append(best_bw)
                test_f1_list.append(f1)

                print(dataset+str(i), 
                      'best frame rate={}, best resolution={} ==> tested f1={}'.format(best_fr/frame_rate, best_resol, f1))
                f.write(dataset + '_' + str(i) + ',' + str(best_resol) + 
                        ',' + str(f1) + ',' + str(best_fr/frame_rate) + ',' + 
                        str(best_bw) + '\n')
            # if test_bw_list and test_f1_list:
            #     plt.scatter(test_bw_list, test_f1_list, label=dataset)
    # plt.xlabel('Bandwidth(Mbps)')
    # plt.xlim(0, 1)
    # plt.ylabel('F1 Score')
    # plt.ylim(0,1)
    # plt.title("Awstream Motivation")
    # plt.legend()
    # plt.savefig('/home/zxxia/figs/awstream/awstream_motivation.png')
    # plt.show()

if __name__ == '__main__':
    main()



##########code below save as alternative implementation of some functions#######

# def compress_images_to_video(path: str, start_frame: int, nb_frames: int, 
#                              frame_rate, resolution,
#                              quality: int, output_name: str):
#     '''
#     Compress a set of frames to a video.
#     start frame: input image frame start index
#     '''
#     cmd = ['ffmpeg', '-r', str(frame_rate), '-f', 'image2', '-s', str(resolution), 
#            '-start_number', str(start_frame), '-i', '{}/%06d.jpg'.format(path), 
#            '-vframes', str(nb_frames), '-vcodec', 'libx264', '-crf', str(quality),
#            '-pix_fmt', 'yuv420p', '-hide_banner', output_name]
#     subprocess.run(cmd)#, stdout=subprocess.PIPE).stdout.decode('utf-8').rstrip()

# def compute_video_size(dataset, start, end, 
#     target_frame_rate, frame_rate, resolution):
#     metadata = load_metadata(PATH + dataset + '/metadata.json')
# 
#     img_path = PATH + dataset
#     frame_array = []
#     img_cn = 0
# 
#     sample_rate = frame_rate/target_frame_rate
#     compress_images_to_video(PATH+dataset, list(range(start,end+1)), target_frame_rate, resolution, 25, 'tmp.mp4')
#     for img_index in range(start, end + 1):
#         # based on sample rate, decide whether this frame is sampled
#         if img_index%sample_rate >= 1:
#             continue
#         else:
#             if resolution == 'original':
#                 filename = img_path + '/' + format(img_index, '06d') + '.jpg'
#                 image_resolution = metadata['resolution']
#             else:
#                 filename = img_path + '/' + resolution + '/' + \
#                 format(img_index+1, '06d') + '.jpg'
#                 image_resolution = IMAGE_RESOLUTION_DICT[resolution]
#         img = cv2.imread(filename)
#         height, width, layers = img.shape
#         assert height == image_resolution[1] and width == image_resolution[0], print(filename, height, width)
# 
#         frame_array.append(img)
#     #print(target_frame_rate, image_resolution, len(frame_array))
#     print(len(frame_array))
#     out = cv2.VideoWriter('tmp.mp4', cv2.VideoWriter_fourcc(*'mp4v'), int(target_frame_rate), (image_resolution[0], image_resolution[1]))
# 
#     for i in range(len(frame_array)):
#         # writing to a image array
#         out.write(frame_array[i])
#     out.release()
#     video_size = os.path.getsize("tmp.mp4")
#     print('target frame rate={}, target image resolution={}. video size={}'.format(target_frame_rate, image_resolution, video_size))
#     return video_size1


