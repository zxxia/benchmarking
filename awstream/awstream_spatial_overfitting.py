''' This script is to used to compute spatial overfitting result of each video.
They are used to draw features vs awstream bandwidth
'''
from collections import defaultdict
import argparse
import os
import pdb
from awstream.profiler import VideoConfig, profile, profile_eval, \
    select_best_config
from constants import RESOL_DICT, CAMERA_TYPES, COCOLabels
from utils.model_utils import load_full_model_detection, \
    filter_video_detections, remove_overlappings
from utils.utils import load_metadata


# PATH = '/mnt/data/zhujun/dataset/Youtube/'
PATH = '/data/zxxia/benchmarking/results/videos/'
DATA_PATH = '/data/zxxia/videos/'
# DATA_PATH = '/data2/zxxia/videos/'
TEMPORAL_SAMPLING_LIST = [1]

OFFSET = 0
RESOLUTION_LIST = ['720p', '540p', '480p', '360p']  # '2160p', '1080p',
ORIGINAL_REOSL = '720p'


def parse_args():
    """ parse args """
    parser = argparse.ArgumentParser(
        description="Awstream with spatial overfitting")
    parser.add_argument("--video", type=str, required=True, help="video name")
    parser.add_argument("--output", type=str, required=True,
                        help="output file")
    parser.add_argument("--log", type=str, required=True, help="log file")
    parser.add_argument("--short_video_length", type=int, required=True,
                        help="short video length in second")
    parser.add_argument("--profile_length", type=int, required=True,
                        help="profile length in second")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    with open(args.output, 'w', 1) as f_out:
        f_out.write('dataset,best_resolution,f1,frame_rate,bandwidth\n')
        f_profile = open(args.log, 'w', 1)
        f_profile.write('dataset,resolution,sample_rate,f1,tp,fp,fn\n')
        metadata = load_metadata(os.path.join(DATA_PATH, args.video,
                                              'metadata.json'))
        resolution = metadata['resolution']
        original_resol = ORIGINAL_REOSL
        # load detection results of fasterRCNN + full resolution +
        # highest frame rate as ground truth
        fps = metadata['frame rate']
        frame_cnt = metadata['frame count']
        num_of_short_videos = frame_cnt//(args.short_video_length*fps)

        dt_dict = defaultdict(None)
        img_path_dict = defaultdict(None)
        img_path_dict[original_resol] = os.path.join(DATA_PATH, args.video,
                                                     original_resol)
        for resol in RESOLUTION_LIST:
            cur_w, cur_h = RESOL_DICT[resol]
            if cur_h > resolution[1]:
                continue
            img_path_dict[resol] = os.path.join(DATA_PATH, args.video, resol)
            dt_file = os.path.join(PATH, args.video, resol, 'profile',
                                   'updated_gt_FasterRCNN_COCO_no_filter.csv')
            print('loading {}...'.format(dt_file))
            dt_dict[resol], frame_cnt = load_full_model_detection(dt_file)
            if args.video in CAMERA_TYPES['moving']:
                dt_dict[resol] = \
                    filter_video_detections(dt_dict[resol],
                                            target_types={COCOLabels.CAR.value,
                                                          COCOLabels.BUS.value,
                                                          # COCOLabels.TRAIN.value,
                                                          COCOLabels.TRUCK.value},
                                            height_range=(cur_h//20, cur_h))
            else:
                dt_dict[resol] = \
                    filter_video_detections(dt_dict[resol], target_types={
                        COCOLabels.CAR.value,
                        COCOLabels.BUS.value,
                        # COCOLabels.TRAIN.value,
                        COCOLabels.TRUCK.value},
                    width_range=(0, cur_w/2),
                    height_range=(cur_h//20, cur_h/2))
            for frame_idx, boxes in dt_dict[resol].items():
                for box_idx, _ in enumerate(boxes):
                    # Merge all cars and trucks into cars
                    dt_dict[resol][frame_idx][box_idx][4] = COCOLabels.CAR.value
                dt_dict[resol][frame_idx] = remove_overlappings(boxes, 0.3)

            if args.video == 'road_trip':
                for frame_idx in dt_dict[resol]:
                    tmp_boxes = []
                    for box in dt_dict[resol][frame_idx]:
                        xmin, ymin, xmax, ymax = box[:4]
                        if ymin >= 500/720*RESOL_DICT[resol][1] \
                                and ymax >= 500/720*RESOL_DICT[resol][1]:
                            continue
                        if (xmax - xmin) >= 2/3 * RESOL_DICT[resol][0]:
                            continue
                        tmp_boxes.append(box)
                    dt_dict[resol][frame_idx] = tmp_boxes

        for i in range(num_of_short_videos):
            clip = args.video + '_' + str(i)
            start_frame = i*args.short_video_length*fps+1+OFFSET*fps
            end_frame = (i+1)*args.short_video_length*fps+OFFSET*fps
            print('{} start={} end={}'.format(clip, start_frame, end_frame))
            # use 30 seconds video for profiling
            profile_start = start_frame
            profile_end = start_frame + fps * args.profile_length - 1

            original_config = VideoConfig(RESOL_DICT[original_resol], fps)

            configs = profile(clip, dt_dict, original_config,
                              [profile_start, profile_end], f_profile,
                              RESOLUTION_LIST, TEMPORAL_SAMPLING_LIST)

            # best_config, best_bw = select_best_config(clip, img_path_dict,
            #                                           original_config, configs,
            #                                           [profile_start,
            #                                            profile_end])
            #
            # print("Profile {}: best resol={}, best fps={}, best bw={}"
            #       .format(clip, best_config.resolution, best_config.fps,
            #               best_bw))
            #
            # test_start = profile_start
            # test_end = profile_end
            #
            # dets = dt_dict[str(best_config.resolution[1]) + 'p']
            # # original_config.debug_print()
            # # best_config.debug_print()
            #
            # f1_score, relative_bw = profile_eval(clip, img_path_dict,
            #                                      dt_dict[original_resol],
            #                                      dets, original_config,
            #                                      best_config,
            #                                      [test_start, test_end])
            #
            # print('{} best fps={}, best resolution={} ==> tested f1={}'
            #       .format(clip, best_config.fps/fps,
            #               best_config.resolution, f1_score))
            # f_out.write(','.join([clip, str(best_config.resolution[1]) + 'p',
            #                       str(f1_score), str(best_config.fps),
            #                       str(relative_bw)]) + '\n')


if __name__ == '__main__':
    main()
