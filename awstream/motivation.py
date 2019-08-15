from awstream.profiler import profile, profile_eval, select_best_config
from collections import defaultdict
from matplotlib import cm
from utils.model_utils import load_full_model_detection_new, eval_single_image
from utils.utils import interpolation, compute_f1, load_metadata
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import subprocess
import sys

# PATH = '/mnt/data/zhujun/new_video/'
PATH = '/mnt/data/zhujun/dataset/Youtube/'
TEMPORAL_SAMPLING_LIST = [1] #[20,15,10,5,4,3,2.5,2,1.8,1.5,1.2,1]
DATASET_LIST = sorted(['motorway'])#sorted(['traffic', 'jp_hw', 'russia', 'tw_road',
               #        'tw_under_bridge', 'highway_normal_traffic', 'nyc',
               #        'lane_split', 'tw', 'tw1', 'jp', 'russia1','drift',
               #        'park', 'walking', 'highway', 'crossroad2', 'crossroad',
               #        'crossroad3', 'crossroad4', 'driving1', 'driving2',
               #        'driving_downtown', 'block', 'block1', 'motorway'])


IMAGE_RESOLUTION_DICT = {'360p': [640, 360],
                         '480p': [854, 480],
                         '540p': [960, 540],
                         '576p': [1024, 576],
                         '720p': [1280, 720],
                         '1080p': [1920, 1080]
                        }

SHORT_VIDEO_LENGTH = 1*60 #seconds
IOU_THRESH = 0.5
TARGET_F1 = 0.9
PROFILE_LENGTH = 30 #seconds
OFFSET = 1*60+30
RESOLUTION_LIST = ['720p', '540p', '480p', '360p'] #


def main():

    with open('awstream_motivation_result_motorway.csv', 'w') as f:
        f.write('dataset,best_resolution,f1,frame_rate,bandwidth\n')
        for dataset in DATASET_LIST:
            f_profile = open('awstream_motivation_profile_{}.csv'.format(dataset), 'w')
            metadata = load_metadata(PATH + dataset + '/metadata.json')
            height = metadata['resolution'][1]
            original_resol = str(height) + 'p'
            # load detection results of fasterRCNN + full resolution +
            #highest frame rate as ground truth
            frame_rate = metadata['frame rate']
            frame_cnt = metadata['frame count']
            num_of_short_videos = 1 #frame_cnt//(SHORT_VIDEO_LENGTH*frame_rate)

            gt_dict = defaultdict(None)
            dt_dict = defaultdict(None)

            for resol in RESOLUTION_LIST:
                gt_file = PATH + dataset + '/' + resol+ '/profile_new/updated_gt_FasterRCNN_COCO.csv'
                if original_resol == resol:
                    dt_file = PATH + dataset + '/' + resol + '/profile/updated_gt_FasterRCNN_COCO.csv'
                else:
                    dt_file = PATH + dataset + '/' + resol + '/profile/gt_{}.csv'.format(resol)
                gt_dict[resol] = load_full_model_detection_new(gt_file)[0]
                dt_dict[resol] = load_full_model_detection_new(dt_file)[0]


            print('Processing', dataset)
            test_bw_list = list()
            test_f1_list = list()
            for i in range(num_of_short_videos):
                start_frame = i * (SHORT_VIDEO_LENGTH*frame_rate) + 1 + OFFSET * frame_rate
                end_frame = (i+1) * (SHORT_VIDEO_LENGTH*frame_rate) + OFFSET * frame_rate
                print('short video start={} end={}'.format(start_frame, end_frame))
                # use 30 seconds video for profiling
                #pdb.set_trace()
                profile_start_frame = start_frame
                profile_end_frame = start_frame + frame_rate * PROFILE_LENGTH - 1
                configs = profile(gt_dict, dt_dict, frame_rate, profile_start_frame,
                                  profile_end_frame, f_profile, RESOLUTION_LIST,
                                  TEMPORAL_SAMPLING_LIST)
                img_path_dict = defaultdict(None)
                for resol in configs.keys():
                    img_path_dict[resol] = PATH + dataset + '/' + resol + '/'

                best_config = select_best_config(img_path_dict, frame_rate,
                                                 profile_start_frame,
                                                 profile_end_frame,
                                                 configs, original_resol)

                print("Finished profiling on frame [{},{}].".format(profile_start_frame, profile_end_frame))
                print("best resol = {}, best frame rate = {}, best bw = {}".format(best_config['resolution'], best_config['frame rate'], best_config['relative bandwidth']))

                # test on the whole video
                #pdb.set_trace()
                test_start_frame = profile_end_frame + 1
                #test_end_frame = end_frame
                test_end_frame = test_start_frame + 240 - 1
                best_resol = best_config['resolution']
                best_resol = '540p' # TODO: need to be removed
                best_fps = best_config['frame rate']
                gt = gt_dict[best_resol]
                dt = dt_dict[best_resol]

                original_config = {'resolution' : original_resol,
                                   'frame rate' : frame_rate}
                f1, relative_bw = profile_eval(img_path_dict, gt, dt,
                                               original_config, best_config,
                                               test_start_frame, test_end_frame)

                print("Finished testing on frame [{},{}].".format(test_start_frame, test_end_frame))
                test_bw_list.append(relative_bw)
                test_f1_list.append(f1)

                print(dataset+str(i),
                      'best frame rate={}, best resolution={} ==> tested f1={}'.format(best_fps/frame_rate, best_resol, f1))
                f.write(dataset + '_' + str(i) + ',' + str(best_resol) +
                        ',' + str(f1) + ',' + str(best_fps) + ',' +
                        str(relative_bw) + '\n')
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


