"""Resize the video and extract the images."""
import argparse
import os
from benchmarking.constants import RESOL_DICT
import subprocess


def resize_video(video_in, video_out, target_size, target_qp=23):
    """Resize the video_in to video_out with target size."""
    cmd = ["ffmpeg", "-y", "-i", video_in, "-hide_banner", "-vf",
           "scale=" +
           target_size, "-crf", str(target_qp), "-vcodec", "libx264",
           "-max_muxing_queue_size", "1024", video_out]
    # \ .format(, target_size, target_qp, video_out)]
    print(cmd)
    subprocess.run(cmd, check=True)


def extract_frames(video, output_path):
    """Extract frames from videos."""
    output_img_name = os.path.join(output_path, "%06d.jpg")
    cmd = ["ffmpeg", "-y", "-i", video, output_img_name, "-qscale:v", "2.0",
           "-hide_banner"]
    print(cmd)
    subprocess.run(cmd, check=True)


def main():
    """Resize videos."""
    parser = argparse.ArgumentParser(description="resize the input video to"
                                     "target resolution")
    parser.add_argument("--input_video", type=str, required=True,
                        help="input video")
    parser.add_argument("--output_video", type=str, required=True,
                        help="output video")
    parser.add_argument("--output_image_path", type=str, default=None,
                        help="output image path")
    parser.add_argument("--resol", type=str, required=True,
                        help="target resolution")
    parser.add_argument("--qp", type=int, default=23,
                        help="quality parameter")
    args = parser.parse_args()
    # path = args.path
    orig_video = args.input_video
    resized_video = args.output_video
    resol_name = args.resol
    target_size = RESOL_DICT[resol_name]
    resized_path = args.output_image_path

    assert args.qp >= 0 and args.qp <= 51
    resize_video(orig_video, resized_video,
                 str(target_size[0])+':'+str(target_size[1]), args.qp)
    # extract frames from the resized videos
    if resized_path is not None:
        extract_frames(resized_video, resized_path)


if __name__ == '__main__':
    main()
