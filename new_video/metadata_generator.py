import json
import os
import argparse
import subprocess

def get_resolution(video):
    cmd = ['ffprobe', '-v', '0', '-of', 'csv=p=0', '-select_streams', 'v:0', 
           '-show_entries', 'stream=width', video]
    width = subprocess.run(cmd, stdout=subprocess.PIPE) \
                      .stdout.decode('utf-8').rstrip()
    cmd = ['ffprobe', '-v', '0', '-of', 'csv=p=0', '-select_streams', 'v:0', 
           '-show_entries', 'stream=height', video]
    height = subprocess.run(cmd, stdout=subprocess.PIPE) \
                       .stdout.decode('utf-8').rstrip()
    return int(width), int(height)

def get_frame_rate(video):
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
    return int(frame_cnt)

def get_duration(video):
    cmd = ['ffprobe', '-v', '0', '-of', 'csv=p=0', '-select_streams', 'v:0', 
           '-show_entries', 'stream=duration', video]
    duration = subprocess.run(cmd, stdout=subprocess.PIPE) \
                         .stdout.decode('utf-8').rstrip()
    return float(duration)

def get_bit_rate(video):
    cmd = ['ffprobe', '-v', '0', '-of', 'csv=p=0', '-select_streams', 'v:0', 
           '-show_entries', 'stream=bit_rate', video]
    bit_rate = subprocess.run(cmd, stdout=subprocess.PIPE) \
                         .stdout.decode('utf-8').rstrip()
    return float(bit_rate)

def get_pixel_format(video):
    cmd = ['ffprobe', '-v', '0', '-of', 'csv=p=0', '-select_streams', 'v:0', 
           '-show_entries', 'stream=pix_fmt', video]
    pix_fmt = subprocess.run(cmd, stdout=subprocess.PIPE) \
                         .stdout.decode('utf-8').rstrip()
    return pix_fmt

def get_level(video):
    cmd = ['ffprobe', '-v', '0', '-of', 'csv=p=0', '-select_streams', 'v:0', 
           '-show_entries', 'stream=level', video]
    level = subprocess.run(cmd, stdout=subprocess.PIPE) \
                         .stdout.decode('utf-8').rstrip()
    return level

def get_codec_name(video):
    cmd = ['ffprobe', '-v', '0', '-of', 'csv=p=0', '-select_streams', 'v:0', 
           '-show_entries', 'stream=codec_name', video]
    codec_name = subprocess.run(cmd, stdout=subprocess.PIPE) \
                         .stdout.decode('utf-8').rstrip()
    return codec_name



def main():
    parser = argparse.ArgumentParser(description="Generate the metadata of a video in json format")
    # group = parser.add_mutually_exclusive_group()
    # group.add_argument("-v", "--verbose", action="store_true")
    # group.add_argument("-q", "--quiet", action="store_true")
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

if __name__=='__main__':
    main()
