import argparse
import cv2
import glob
import logging
import json
import os
import time
from benchmarking.constants import babygroot_DT_ROOT
from benchmarking.ground_truth.run_inference import run_inference

def read_video_info(video_path):
    video_info = {}
    vid = cv2.VideoCapture(video_path)
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(vid.get(cv2.CAP_PROP_FPS))      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps
    minutes = int(duration/60)
    seconds = duration%60
    vid.release()
    video_info['video_name'] = os.path.basename(video_path).replace('.mp4', '')
    video_info['duration'] = str(minutes) + ':' + str(seconds)
    video_info['resol'] = [width, height]
    video_info['frame_rate'] = fps
    video_info['frame_count'] = frame_count
    return video_info

def read_image_info(img_files, frame_rate):
    image_info = {}
    image_info['frame_count'] = len(img_files)
    image_info['frame_rate'] = frame_rate

    img = cv2.imread(img_files[0])
    height, width, channels = img.shape

    duration = image_info['frame_count']/frame_rate
    minutes = int(duration/60)
    seconds = duration%60
    image_info['duration'] = str(minutes) + ':' + str(seconds)
    image_info['resol'] = [width, height]
    return image_info

def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(
        description="VideoStorm with temporal overfitting")
    parser.add_argument("--gpu_num", type=str, required=True, default="3")
    parser.add_argument("--name", type=str, required=True, help="dataset name")
    parser.add_argument("--type", type=str, required=True, default="mp4")
    parser.add_argument("--camera", type=str, required=True, default=None,
                        help="camera type")
    parser.add_argument("--frame_rate", type=int, default=0,
                        help="frame rate")
    args = parser.parse_args()
    return args

def preprocessing(path, _type, frame_rate, camera):
    if _type == 'video':
        mp4_files = glob.glob(path + '/*.mp4')
        assert len(mp4_files) == 1, logging.ERROR('No mp4 found.')
        dataset_info = read_video_info(mp4_files[0])

    else:
        img_files = glob.glob(path + '/*.' + _type)
        assert len(img_files) >= 1, logging.ERROR('No images found.')
        assert frame_rate != 0, print(frame_rate)
        dataset_info = read_image_info(img_files, frame_rate)

    dataset_info['type'] = _type
    dataset_info['camera_type'] = camera
    dataset_info['path'] = path
    return dataset_info



def main():
    args = parse_args()
    # define video path, name
    # video source format
    # path = os.path.join(babygroot_DT_ROOT, args.name)


    waymo_path = '/mnt/data/zhujun/dataset/Waymo/waymo_images'
    folder_list = sorted([x for x in os.listdir(waymo_path) if os.path.isdir(os.path.join(waymo_path, x))])
    args.type = 'jpg'
    args.camera = 'moving'
    args.frame_rate = 10
    for folder in folder_list[0:1]:
        path = os.path.join(waymo_path, folder, 'FRONT')
        print(path)
        logging.basicConfig(filename=path + '/run_all.log', filemode='w', level=logging.DEBUG)
        start_time = time.time()
        # generate basic metadata
        logging.info('Processing dataset: %s', path)
        dataset_info = preprocessing(path, args.type, args.frame_rate, args.camera)
        logging.info(json.dumps(dataset_info))

        # run inference using multiple models 
        run_inference(dataset_info, args.gpu_num)
        

    

    print(time.time() - start_time)

if __name__ == '__main__':
    main()