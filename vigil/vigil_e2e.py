""" vigil overfitting scirpt  """
import argparse
import csv
import pdb
import os
from benchmarking.video import YoutubeVideo
from benchmarking.vigil.Vigil import Vigil, mask_video, mask_video_ffmpeg

PATH = '/data/zxxia/videos/'

OFFSET = 0  # The time offset from the start of the video. Unit: seconds

DT_ROOT = '/data/zxxia/benchmarking/results/videos'
DATA_ROOT = '/data/zxxia/videos'


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="vigil")
    # parser.add_argument("--path", type=str, help="path contains all datasets")
    parser.add_argument("--video", type=str, help="video name")
    parser.add_argument("--metadata", type=str, default='',
                        help="metadata file in Json")
    parser.add_argument("--output", type=str, help="output result file")
    # parser.add_argument("--log", type=str, help="log middle file")
    parser.add_argument("--profile_length", type=int,
                        help="profile video length in seconds")
    parser.add_argument("--short_video_length", type=int,
                        help="short video length in seconds")
    parser.add_argument("--offset", type=int,
                        help="offset from beginning of the video in seconds")
    parser.add_argument("--save_path", type=str,
                        help="output video saving path")
    args = parser.parse_args()
    return args


def main():
    """Vigil end-to-end."""
    args = parse_args()
    dt_file = os.path.join(
        DT_ROOT, args.video, '720p',
        'profile/updated_gt_Inception_COCO_no_filter.csv')
    img_path = os.path.join(DATA_ROOT, args.video, '720p')
    video = YoutubeVideo(args.video, '720p', args.metadata,
                         dt_file, img_path, filter_flag=True)
    output_path = '/data/zxxia/benchmarking/Vigil/masked_images/'+args.video
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    # mask_video_ffmpeg(video, 0.2, 0.2, save_path=output_path)
    # return
    # mask_video(video, 0.2, 0.2,
    #            save_path='/data/zxxia/benchmarking/Vigil/test_bg')

    # load pipeline
    pipeline = Vigil()
    # Load groud truth
    dt_file = os.path.join(
        DT_ROOT, args.video, '720p',
        'profile/updated_gt_FasterRCNN_COCO_no_filter.csv')
    img_path = os.path.join(DATA_ROOT, args.video, '720p')
    original_video = YoutubeVideo(args.video, '720p', args.metadata,
                                  dt_file, img_path, filter_flag=True)

    # # Load fastercnn detections on blacked background images
    # dt_file = '/data/zxxia/blackbg/'+args.video+'/' + \
    #     'profile/updated_gt_FasterRCNN_COCO_no_filter.csv'

    dt_file = os.path.join(
        DT_ROOT, args.video, '720p',
        'profile/updated_gt_Inception_COCO_no_filter.csv')
    # img_path = '/data/zxxia/blackbg/'+args.video+'/'
    img_path = '/data/zxxia/benchmarking/Vigil/masked_images/' + args.video
    video = YoutubeVideo(args.video, '720p',
                         args.metadata, dt_file, img_path, filter_flag=True)

    # Load haar detection results
    # haar_dt_file = 'haar_detections_new/haar_{}.csv'.format(dataset)
    # haar_dets = load_haar_detection(haar_dt_file)
    # haar_dets = filter_haar_detection(haar_dets, height_range=(720//20, 720))

    nb_short_videos = original_video.frame_count//(args.short_video_length *
                                                   original_video.frame_rate)
    profile_frame_cnt = args.profile_length * original_video.frame_rate
    # do the video bandwidth computation
    # return
    with open(args.output, 'w', 1) as f_out:
        writer = csv.writer(f_out)
        writer.writerow(
            ['video', 'bw', 'f1'])
        for i in range(nb_short_videos):
            clip = args.video + '_' + str(i)
            start_frame = i*args.short_video_length*original_video.frame_rate+1
            end_frame = (i+1) * args.short_video_length * \
                original_video.frame_rate
            profile_start_frame = start_frame
            profile_end_frame = profile_start_frame + profile_frame_cnt - 1
            test_start_frame = profile_end_frame + 1
            test_end_frame = end_frame
            bw, f1 = pipeline.evaluate(clip, video, original_video,
                                       [test_start_frame, test_end_frame],
                                       args.save_path)

            print('{}, start={}, end={}, f1={}, bw={}'
                  .format(clip, start_frame, end_frame, f1, bw))
            writer.writerow([clip, bw, f1])
            # str(np.mean(relative_up_areas)),
            # str(np.mean(tot_obj_areas))+'\n']))


if __name__ == '__main__':
    main()
