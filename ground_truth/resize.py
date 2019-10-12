""" This script is used to resize the video and extract the images """
import argparse
import os
from constants import RESOL_DICT


def resize_video(video_in, video_out, target_size):
    """resize the video_in to video_out with target size"""
    cmd = "ffmpeg -y -i {} -hide_banner -vf scale={}"\
          " -max_muxing_queue_size 1024 {}"\
          .format(video_in, target_size, video_out)
    print(cmd)
    os.system(cmd)


def extract_frames(video, output_path):
    """extract frames from videos"""
    cmd = "ffmpeg -y -i {} {}%06d.jpg -qscale:v 2.0 -hide_banner"\
          .format(video, output_path)
    print(cmd)
    os.system(cmd)


def main():
    """resize videos"""
    parser = argparse.ArgumentParser(description="resize the input video to"
                                     "target resolution")
    parser.add_argument("--input_video", type=str, help="input video")
    parser.add_argument("--output_video", type=str, help="output video")
    parser.add_argument("--output_image_path", type=str,
                        help="output image path")
    parser.add_argument("--resol", type=str, help="target resolution")
    # parser.add_argument("--metadata", type=str, default='',
    #                     help="metadata file in Json")
    args = parser.parse_args()
    # path = args.path
    orig_video = args.input_video
    resized_video = args.output_video
    resol_name = args.resol
    target_size = RESOL_DICT[resol_name]
    resized_path = args.output_image_path

    resize_video(orig_video, resized_video,
                 str(target_size[0])+':'+str(target_size[1]))
    # extract frames from the resized videos
    extract_frames(resized_video, resized_path)


if __name__ == '__main__':
    main()
