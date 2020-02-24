"""Resize the video and extract the images."""
import argparse
import subprocess


def resize_video(video_in, video_out, target_size, target_qp=23):
    """Resize the video_in to video_out with target size."""
    cmd = "ffmpeg -y -i {} -hide_banner -vf scale={} -crf {}" \
        " -vcodec libx264 -max_muxing_queue_size 1024 {}".format(
            video_in, target_size, target_qp, video_out)
    print(cmd)
    subprocess.run(cmd.split(' '), check=True)


def resize_image(image_in, image_out, target_size, target_qp=23):
    """Resize the image_in to image_out with target size."""
    cmd = "ffmpeg - y - i {} - vf scale = {} -crf {} {}".format(
        image_in, target_size, target_qp, image_out)
    print(cmd)
    subprocess.run(cmd.split(' '), check=True)


def main():
    """Resize videos."""
    parser = argparse.ArgumentParser(description="Resize the input video to"
                                     "target resolution.")
    parser.add_argument("--input_video", type=str, required=True,
                        help="input video")
    parser.add_argument("--output_video", type=str, required=True,
                        help="output video")
    parser.add_argument("--target_width", type=str, required=True,
                        help="target width")
    parser.add_argument("--target_height", type=str, required=True,
                        help="target height")
    parser.add_argument("--qp", type=int, default=23,
                        help="quality parameter")
    args = parser.parse_args()
    # path = args.path
    orig_video = args.input_video
    resized_video = args.output_video

    assert args.qp >= 0 and args.qp <= 51
    resize_video(orig_video, resized_video,
                 str(args.target_width)+':'+str(args.target_height), args.qp)


if __name__ == '__main__':
    main()
