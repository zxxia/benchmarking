"""Extract the images."""
import argparse
import os
import subprocess


def extract_frames(video, output_path, qscale):
    """Extract frames from videos."""
    if not os.path.exists(output_path):
        print(f'mkdir {output_path}')
        os.mkdir(output_path)
    output_img_name = os.path.join(output_path, "%06d.jpg")
    cmd = "ffmpeg -y -i {} -qscale:v {} {} -hide_banner".format(
        video, qscale, output_img_name)
    print(cmd)
    subprocess.run(cmd.split(' '), check=True)


def main():
    """Resize videos."""
    parser = argparse.ArgumentParser(
        description="Extract frames from a video into jpg files.")
    parser.add_argument("--input_video", type=str, required=True,
                        help="input video")
    parser.add_argument("--output_image_path", type=str, default=None,
                        help="output image path")
    parser.add_argument("--qscale", type=float, default=1,
                        help="qscale in ffmpeg. Range: 1-31. Default: 1.")
    args = parser.parse_args()
    video = args.input_video

    # extract frames from the input video
    extract_frames(video, args.output_image_path, args.qscale)


if __name__ == '__main__':
    main()
