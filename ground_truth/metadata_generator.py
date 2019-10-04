import json
import os
import argparse
import subprocess


def get_resolution(video):
    file_extension = os.path.splitext(video)[-1]
    if file_extension == '.ts':
        width_cmd = ['ffprobe', '-v', '0', '-of', 'csv=p=0', '-select_streams', 'v:0',
           '-show_entries', 'program_stream=width', video]
        height_cmd = ['ffprobe', '-v', '0', '-of', 'csv=p=0', '-select_streams', 'v:0',
           '-show_entries', 'program_stream=height', video]
    else:
        width_cmd = ['ffprobe', '-v', '0', '-of', 'csv=p=0', '-select_streams', 'v:0',
           '-show_entries', 'stream=width', video]
        height_cmd = ['ffprobe', '-v', '0', '-of', 'csv=p=0', '-select_streams', 'v:0',
           '-show_entries', 'stream=height', video]
    height = subprocess.run(height_cmd, stdout=subprocess.PIPE) \
                       .stdout.decode('utf-8').rstrip()
    width = subprocess.run(width_cmd, stdout=subprocess.PIPE) \
                      .stdout.decode('utf-8').rstrip()

    try:
        width = int(width)
        height = int(height)
    except ValueError:
        width = 'N/A'
        height = 'N/A'
    return width, height


def get_frame_rate(video):
    file_extension = os.path.splitext(video)[-1]
    if file_extension == '.ts':
        cmd = ['ffprobe', '-v', '0', '-of', 'csv=p=0', '-select_streams', 'v:0',
           '-show_entries', 'program_stream=r_frame_rate', video]
    else:
        cmd = ['ffprobe', '-v', '0', '-of', 'csv=p=0', '-select_streams', 'v:0',
           '-show_entries', 'stream=r_frame_rate', video]
    a, b = subprocess.run(cmd, stdout=subprocess.PIPE) \
                       .stdout.decode('utf-8').rstrip().split('/')
    frame_rate = round(float(a)/float(b))
    return int(frame_rate)


def get_frame_count(video):
    cmd = ['ffprobe', '-v', '0', '-of', 'csv=p=0', '-select_streams', 'v:0',
           '-show_entries', 'stream=nb_frames', video]
    frame_cnt = subprocess.run(cmd, stdout=subprocess.PIPE) \
                          .stdout.decode('utf-8').rstrip()
    try:
        frame_cnt = int(frame_cnt)
    except ValueError:
        frame_cnt = "N/A"

    return frame_cnt


def get_duration(video):
    file_extension = os.path.splitext(video)[-1]
    if file_extension == '.ts':
        cmd = ['ffprobe', '-v', '0', '-of', 'csv=p=0', '-select_streams', 'v:0',
           '-show_entries', 'program_stream=duration', video]
    else:
        cmd = ['ffprobe', '-v', '0', '-of', 'csv=p=0', '-select_streams', 'v:0',
           '-show_entries', 'stream=duration', video]
    duration = subprocess.run(cmd, stdout=subprocess.PIPE) \
                         .stdout.decode('utf-8').rstrip()
    return float(duration)


def get_bit_rate(video):
    file_extension = os.path.splitext(video)[-1]
    if file_extension == '.mp4':
        cmd = ['ffprobe', '-v', '0', '-of', 'csv=p=0', '-select_streams', 'v:0',
           '-show_entries', 'stream=bit_rate', video]
    else:
        cmd = ['ffprobe', '-v', '0', '-of', 'csv=p=0', '-select_streams', 'v:0',
               '-show_entries', 'format=bit_rate', video]
    bit_rate = subprocess.run(cmd, stdout=subprocess.PIPE) \
                         .stdout.decode('utf-8').rstrip()

    return float(bit_rate)


def get_pixel_format(video):
    file_extension = os.path.splitext(video)[-1]
    if file_extension == '.ts':
        cmd = ['ffprobe', '-v', '0', '-of', 'csv=p=0', '-select_streams', 'v:0',
           '-show_entries', 'program_stream=pix_fmt', video]
    else:
        cmd = ['ffprobe', '-v', '0', '-of', 'csv=p=0', '-select_streams', 'v:0',
           '-show_entries', 'stream=pix_fmt', video]
    pix_fmt = subprocess.run(cmd, stdout=subprocess.PIPE) \
                         .stdout.decode('utf-8').rstrip()
    return pix_fmt


def get_level(video):
    file_extension = os.path.splitext(video)[-1]
    if file_extension == '.ts':
        cmd = ['ffprobe', '-v', '0', '-of', 'csv=p=0', '-select_streams', 'v:0',
           '-show_entries', 'program_stream=level', video]
    else:
        cmd = ['ffprobe', '-v', '0', '-of', 'csv=p=0', '-select_streams', 'v:0',
           '-show_entries', 'stream=level', video]
    level = subprocess.run(cmd, stdout=subprocess.PIPE) \
                         .stdout.decode('utf-8').rstrip()
    return level


def get_codec_name(video):
    file_extension = os.path.splitext(video)[-1]
    if file_extension == '.ts':
        cmd = ['ffprobe', '-v', '0', '-of', 'csv=p=0', '-select_streams', 'v:0',
               '-show_entries', 'program_stream=codec_name', video]
    else:
        cmd = ['ffprobe', '-v', '0', '-of', 'csv=p=0', '-select_streams', 'v:0',
               '-show_entries', 'stream=codec_name', video]
    codec_name = subprocess.run(cmd, stdout=subprocess.PIPE) \
                         .stdout.decode('utf-8').rstrip()
    return codec_name


def main():
    parser = argparse.ArgumentParser(description="Generate the metadata of a video in json format")
    parser.add_argument("video", type=str, help="the absolute path of the input video")
    parser.add_argument("output", type=str, help="the absolute path where json file will be generated")
    args = parser.parse_args()

    print(args.video)
    print(args.output)
    metadata = dict()
    metadata['resolution'] = get_resolution(args.video)
    metadata['frame rate'] = get_frame_rate(args.video)
    metadata['duration'] = get_duration(args.video)
    metadata['frame count'] = get_frame_count(args.video)
    metadata['bit rate'] = get_bit_rate(args.video)
    metadata['pixel format'] = get_pixel_format(args.video)
    metadata['level'] = get_level(args.video)
    metadata['codec name'] = get_codec_name(args.video)
    with open(args.output + '/' + "metadata.json", 'w') as f:
        json.dump(metadata, f, sort_keys=True, indent=4)


if __name__ == '__main__':
    main()
