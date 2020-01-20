""" vigil overfitting scirpt  """
import argparse
import csv
import pdb
import os
import sys
sys.path.append('../../')
from benchmarking.video import YoutubeVideo
from benchmarking.vigil.Vigil import Vigil, mask_video_ffmpeg
# mask_video,

# PATH = '/data/zxxia/videos/'

OFFSET = 0  # The time offset from the start of the video. Unit: seconds

DT_ROOT = '/mnt/data/zhujun/dataset/Youtube'
Vigil_DATA_ROOT = '/mnt/data/zhujun/dataset/Vigil_result/blackbg'

SHORT_VIDEO_LENGTH = 30

profile_length = 10
name = 'driving1'
OFFSET = 0
SMALL_MODEL_PATH = '/mnt/data/zhujun/dataset/NoScope_finetuned_models'

def main():
    """Vigil end-to-end."""
    output_path = './'
    # load pipeline
    pipeline = Vigil()
    # Load groud truth
    dt_file = os.path.join(DT_ROOT, name, '720p', 'profile',
                           'updated_gt_FasterRCNN_COCO_no_filter.csv')
    metadata_file = DT_ROOT + '/{}/metadata.json'.format(name)  
    # # Load fastercnn detections on blacked background images

    original_video = YoutubeVideo(name, '720p', metadata_file, dt_file, None)



    dt_file = os.path.join(Vigil_DATA_ROOT, name, 'profile',
                           'updated_gt_FasterRCNN_COCO_no_filter.csv')
    img_path = os.path.join(Vigil_DATA_ROOT, name)
    cropped_video = YoutubeVideo(name, '720p', metadata_file, dt_file, img_path)
    # Load haar detection results
    # haar_dt_file = 'haar_detections_new/haar_{}.csv'.format(dataset)
    # haar_dets = load_haar_detection(haar_dt_file)
    # haar_dets = filter_haar_detection(haar_dets, height_range=(720//20, 720))

    nb_short_videos = original_video.frame_count//(SHORT_VIDEO_LENGTH *
                                                   original_video.frame_rate)
    # profile_frame_cnt = profile_length * original_video.frame_rate
    # do the video bandwidth computation
    # return
    with open(output_path + '/vigil_e2e_result_' + name + '_with_videosize.csv', 'w', 1) as f_out:
        writer = csv.writer(f_out)
        writer.writerow(
            ['video', 'bw', 'f1', 'video_bw'])
        for i in range(nb_short_videos):
            clip = name + '_' + str(i)
            start_frame = i*SHORT_VIDEO_LENGTH * \
                original_video.frame_rate+1+OFFSET*original_video.frame_rate
            end_frame = (i+1)*SHORT_VIDEO_LENGTH * \
                original_video.frame_rate+OFFSET*original_video.frame_rate
            profile_start_frame = start_frame
            profile_end_frame = start_frame + original_video.frame_rate * \
                profile_length - 1
            test_start_frame = profile_end_frame + 1
            test_end_frame = end_frame
            original_video_save_path = os.path.join(
                             SMALL_MODEL_PATH, name, 'data')

            cropped_video_save_path = os.path.join(
                             Vigil_DATA_ROOT, name)
            bw, f1, video_bw = pipeline.evaluate(clip, cropped_video, original_video,
                                       [test_start_frame, test_end_frame],
                                       original_video_save_path,
                                       cropped_video_save_path)

            print('{}, start={}, end={}, f1={}, bw={}, video_bw={}'
                  .format(clip, start_frame, end_frame, f1, bw, video_bw))
            writer.writerow([clip, bw, f1, video_bw])


if __name__ == '__main__':
    main()
