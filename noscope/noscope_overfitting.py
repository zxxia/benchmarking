"""VideoStorm Overfitting Script."""
import argparse
import csv
import pdb
import sys
import os
# from video import YoutubeVideo
import numpy as np
sys.path.append('../../')

from benchmarking.noscope.NoScope import NoScope, load_ground_truth, \
    load_simple_model_classification, train_small_model
from benchmarking.video import YoutubeVideo

THRESH_LIST = np.arange(0.7, 1, 0.1)

OFFSET = 0  # The time offset from the start of the video. Unit: seconds


VIDEOS = ['crossroad', 'crossroad2', 'crossroad3', 'crossroad4','drift',
          'driving1', 'driving_downtown', 'highway',
          'nyc', 'jp',  'lane_split',  'driving2',
          'motorway', 'park', 'russia', 'russia1', 
          'traffic', 'tw', 'tw1',
          'tw_under_bridge']

# VIDEOS = ['cropped_driving2', 'cropped_crossroad4', 'cropped_crossroad4_2', 'cropped_crossroad5' ]
VIDEOS = ['crossroad3', 'crossroad4', 'motorway', 'drift'
]
VIDEOS = ['cropped_driving2']
# DT_ROOT = '/data/zxxia/benchmarking/results/videos'
DT_ROOT = '/mnt/data/zhujun/dataset/Youtube'
SHORT_VIDEO_LENGTH = 30

profile_length = 30

def main():
    """NoScope."""
    gpu_num = '1'
    triggered_frames = []
    # tstamp = 0
    for name in VIDEOS:
        if "cropped" in name:
            resol = '360p'
        else:
            resol = '720p'

        pipeline = NoScope(THRESH_LIST, 'noscope_profile_{}.csv'.format(name))
        metadata_file = DT_ROOT + '/{}/metadata.json'.format(name)
        model = 'FasterRCNN'
        dt_file = os.path.join(
            DT_ROOT, name, resol,
            'profile/updated_gt_FasterRCNN_COCO_no_filter.csv')
        img_path = os.path.join(
            DT_ROOT, name, resol)
        video = YoutubeVideo(name, resol, metadata_file, dt_file, img_path)
        model_save_path = './trained_models/'
        train_small_model(name, gpu_num, model_save_path, video)


    # # video = YoutubeVideo(args.video, '720p', args.metadata_file, args.input,
    # #                      None, True)
    # frame_rate = 30  # , video.frame_rate
    # # for dataset in ['cropped_crossroad4', 'cropped_crossroad4_2',
    # #                 'cropped_crossroad5', 'cropped_driving2']:
    # gt_file = '../model_pruning/label/' + \
    #     args.video + '_ground_truth' + filename + '.csv'
    # gt = load_ground_truth(gt_file)
    # small_model_result = load_simple_model_classification(
    #     '../model_pruning/label/noscope_small_model_predicted_{}{}.csv'.format(args.video, filename))
    # frame_count = len(small_model_result)  # video.frame_count

    # system = NoScope(
    #     THRESH_LIST, 'noscope_profile_{}.csv'.format(args.video))

    # with open('noscope_result_{}.csv'.format(args.video), 'w', 1) as f_out:
    #     f_out.write("video_name,best_thresh,gpu,f1\n")

    #     # Chop long videos into small chunks
    #     # Floor division drops the last sequence of frames which is not as
    #     # long as short_video_length
    #     profile_frame_cnt = args.profile_length * frame_rate
    #     chunk_frame_cnt = args.short_video_length * frame_rate
    #     num_of_chunks = (frame_count-OFFSET*frame_rate)//chunk_frame_cnt

    #     for i in range(num_of_chunks):
    #         clip = args.video + '_' + str(i)
    #         # the 1st frame in the chunk
    #         start_frame = i * chunk_frame_cnt + 1 + OFFSET * frame_rate
    #         # the last frame in the chunk
    #         end_frame = (i + 1) * chunk_frame_cnt + OFFSET * frame_rate
    #         print('short video start={}, end={}'.format(start_frame,
    #                                                     end_frame))
    #         assert args.short_video_length >= args.profile_length

    #         profile_start_frame = start_frame
    #         profile_end_frame = profile_start_frame + profile_frame_cnt - 1
    #         triggered_frames.extend(list(range(profile_start_frame,
    #                                            profile_end_frame + 1)))

    #         best_thresh = system.profile(clip, small_model_result, gt,
    #                                      profile_start_frame,
    #                                      profile_end_frame)
    #         # test on the rest of the short video

    #         # test_start_frame = profile_start_frame
    #         # test_end_frame = profile_end_frame
    #         test_start_frame = profile_end_frame + 1
    #         test_end_frame = end_frame

    #         gpu, f1_score = system.evaluate(clip, small_model_result, gt,
    #                                         test_start_frame, test_end_frame,
    #                                         best_thresh)
    #         # triggered_frames.extend(triggered_frames_tmp)

    #         print(clip, best_thresh, gpu, f1_score)
    #         f_out.write(','.join([clip, str(best_thresh), str(gpu),
    #                               str(f1_score)]) + '\n')

    #     # with open('{}_trace.csv'.format(args.video), 'w', 1) as f_trace:
    #     #     writer = csv.writer(f_trace)
    #     #     writer.writerow(['frame id', 'timestamp', 'trigger'])
    #     #     for i in range(1, frame_count + 1):
    #     #         if i in triggered_frames:
    #     #             writer.writerow([i, tstamp, 1])
    #     #         else:
    #     #             writer.writerow([i, tstamp, 0])
    #     #         tstamp += 1/frame_rate


if __name__ == '__main__':
    main()
