"""Extract the images."""
import argparse
import os
import subprocess


def extract_frames(video, output_path):
    """Extract frames from videos."""
    if not os.path.exists(output_path):
        print(f'mkdir {output_path}')
        os.mkdir(output_path)
    output_img_name = os.path.join(output_path, "%06d.jpg")
    cmd = "ffmpeg -y -i {} {} -qscale:v 2.0 -hide_banner".format(
        video, output_img_name)
    print(cmd)
    subprocess.run(cmd.split(' '), check=True)


def main():
    """Resize videos."""
    parser = argparse.ArgumentParser(description="Resize the input video to"
                                     "target resolution.")
    parser.add_argument("--input_video", type=str, required=True,
                        help="input video")
    parser.add_argument("--output_image_path", type=str, default=None,
                        help="output image path")
    args = parser.parse_args()
    video = args.input_video

    # extract frames from the input video
    extract_frames(video, args.output_image_path)


if __name__ == '__main__':
    main()
