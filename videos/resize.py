"""Resize the video and extract the images."""
import argparse
import glob
import os
import subprocess


def resize_video(video_in, video_out, target_width, target_height, target_qp):
    """Resize the video_in to video_out with target size."""
    cmd = "ffmpeg -y -i {} -hide_banner -vf scale={}:{} -crf {}" \
        " -vcodec libx264 -max_muxing_queue_size 1024 {}".format(
            video_in, target_width, target_height, target_qp, video_out)
    print(cmd)
    subprocess.run(cmd.split(' '), check=True)


def resize_image(image_in, image_out, target_width, target_height, target_qp):
    """Resize the image_in to image_out with target size."""
    if os.path.splitext(image_in)[1] == '.jpg':
        cmd = "ffmpeg -y -i {} -vf scale={}:{} -q:v 2.0 {} -hide_banner" \
            .format(image_in, target_width, target_height, image_out)
    if os.path.splitext(image_in)[1] == '.png':
        cmd = "ffmpeg -y -i {} -vf scale={}:{} {} -hide_banner" .format(
            image_in, target_width, target_height, image_out)
    print(cmd)
    subprocess.run(cmd.split(' '), check=True)


def main():
    """Resize videos."""
    parser = argparse.ArgumentParser(description="Resize the input video to"
                                     "target resolution.")
    parser.add_argument("--input", type=str, required=True,
                        help="input video(e.g. in.mp4) or path to images")
    parser.add_argument("--output", type=str, required=True,
                        help="output video(e.g. out.mp4) or path to images")
    parser.add_argument("--target_width", type=str, required=True,
                        help="target width, e.g. 1280")
    parser.add_argument("--target_height", type=str, required=True,
                        help="target height, e.g. 720")
    parser.add_argument("--qp", type=int, default=23, help="quality parameter"
                        "(-crf in ffmpeg) with range [0, 51].Default: 23. 0 "
                        "means lossless. 51 means the most compresssion.")
    args = parser.parse_args()

    if os.path.splitext(args.input)[1] == '.mp4' and \
            os.path.splitext(args.output)[1] == '.mp4':
        # input and output are videos
        assert args.qp >= 0 and args.qp <= 51, "qp must be in range [0, 51]."
        resize_video(args.input, args.output, args.target_width,
                     args.target_height, args.qp)
    elif os.path.isdir(args.input) and os.path.isdir(args.output):
        # input and output are directories which contain images
        img_paths = sorted(glob.glob(os.path.join(args.input, '*.jpg'))) + \
            sorted(glob.glob(os.path.join(args.input, '*.png')))
        for img_path in img_paths:
            out_path = os.path.join(args.output, os.path.basename(img_path))
            print(img_path, out_path)
            resize_image(img_path, out_path, args.target_width,
                         args.target_height, args.qp)


if __name__ == '__main__':
    main()
