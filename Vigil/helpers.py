""" helpers functions in Vigil """
import os
import subprocess
import numpy as np


def crop_image(img_in, xmin, ymin, xmax, ymax, w_out, h_out, img_out):
    """ Crop an image """
    cmd = ['ffmpeg', '-y', '-hide_banner', '-i', img_in, '-vf',
           'crop={}:{}:{}:{},pad={}:{}:0:0:black'
           .format(xmax-xmin, ymax-ymin, xmin, ymin, w_out, h_out), img_out]

    subprocess.run(cmd)


def compress_images_to_video(list_file: str, frame_rate: str, resolution: str,
                             quality: int, output_name: str):
    """ Compress a set of frames to a video.  """
    cmd = ['ffmpeg', '-y', '-r', str(frame_rate), '-f', 'concat', '-safe', '0',
           '-i', list_file, '-s', str(resolution), '-vcodec', 'libx264',
           '-crf', str(quality), '-pix_fmt', 'yuv420p', '-hide_banner',
           output_name]
    subprocess.run(cmd)


def compute_video_size(video, img_path, start, end, target_frame_rate,
                       frame_rate, image_resolution):
    """ Compress a set of frames to a video and measure the video size.  """
    sample_rate = frame_rate/target_frame_rate
    # Create a tmp list file contains all the selected iamges
    tmp_list_file = video+'_list.txt'
    with open(tmp_list_file, 'w') as f_list:
        for img_index in range(start, end + 1):
            # based on sample rate, decide whether this frame is sampled
            if img_index % sample_rate >= 1:
                continue
            line = 'file \'{}/{:06}.jpg\'\n'.format(img_path, img_index)
            f_list.write(line)

    frame_size = str(image_resolution[0]) + 'x' + str(image_resolution[1])
    tmp_video = video + '_tmp.mp4'
    compress_images_to_video(tmp_list_file, target_frame_rate, frame_size,
                             25, tmp_video)
    video_size = os.path.getsize(tmp_video)
    # os.remove(tmp_video)
    os.remove(tmp_list_file)
    print('target frame rate={}, target image resolution={}. video size={}'
          .format(target_frame_rate, image_resolution, video_size))
    return video_size


def remove_background(image, boxes):
    """ Black out pixels not in boxes """
    mask = np.zeros(image.shape, dtype=np.uint8)
    for box in boxes:
        xmin, ymin, xmax, ymax = box[:4]
        mask[ymin:ymax, xmin:xmax] = 1
    ret = image.copy()
    ret *= mask
    return ret, mask


def convert_box_coordinate(box):
    """ box: x, y, w, h
    return: [xmin, ymin, xmax, ymax]
    """
    xmin = int(box[0])
    ymin = int(box[1])
    xmax = xmin + int(box[2])
    ymax = ymin + int(box[3])
    return [xmin, ymin, xmax, ymax]


# def overlap_percentage(boxA, boxB):
#     """ boxA, boxB: [xmin, ymin, xmax, ymax]
#     """
#     # boxA = convert_box_coordinate(boxA)
#     # boxB = convert_box_coordinate(boxB)
#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[2], boxB[2])
#     yB = min(boxA[3], boxB[3])
#
#     # compute the area of intersection rectangle
#     interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
#     # compute the area of both the prediction and ground-truth
#     # rectangles
#     boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
#     # print(boxA, boxB)
#     # print(xA, yA, xB, yB)
#     return float(interArea)/boxBArea
