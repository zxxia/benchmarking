"""Vigil Implementation."""
import os
import sys
import time
from subprocess import Popen

import cv2
import numpy as np

from evaluation.f1 import compute_f1, evaluate_frame


class Vigil():
    """Vigil Implementation."""

    def __init__(self):
        pass

    def profile(self):
        pass

    def evaluate(self, video_name, video, original_video, frame_range,
                 video_save_path):
        """Evaluate."""
        tpos = {}
        fpos = {}
        fneg = {}
        bw = 0
        for i in range(frame_range[0], frame_range[1] + 1):
            bw += video.get_frame_filesize(i)
            tpos[i], fpos[i], fneg[i] = evaluate_frame(
                original_video.get_frame_detection(i),
                video.get_frame_detection(i))
        tp_total = sum(tpos.values())
        fp_total = sum(fpos.values())
        fn_total = sum(fneg.values())
        f1_score = compute_f1(tp_total, fp_total, fn_total)
        original_bw = original_video.encode(
            os.path.join(video_save_path, video_name + '_original.mp4'),
            list(range(frame_range[0], frame_range[1] + 1)),
            original_video.frame_rate)
        video_bw = video.encode(os.path.join(video_save_path,
                                             video_name+'_cropped.mp4'),
                                list(range(frame_range[0], frame_range[1]+1)),
                                video.frame_rate)
        return bw/original_bw, f1_score, video_bw/original_bw, original_bw


def mask_image(img, boxes):
    """Keep pixels within boxes and change the background into black."""
    mask = np.zeros(img.shape, dtype=np.uint8)
    for box in boxes:
        xmin, ymin, xmax, ymax = box[:4]
        mask[ymin:ymax, xmin:xmax] = 1

    masked_img = img.copy()
    masked_img *= mask
    return mask, masked_img


def mask_video(video, w_delta_percent, h_delta_percent, save_path=None):
    """Keep pixels within boxes and change the background into black."""
    for i in range(video.start_frame_index, video.end_frame_index + 1):
        boxes = video.get_frame_detection(i)
        boxes = resize_bboxes(boxes, w_delta_percent,
                              h_delta_percent, video.resolution)
        img = video.get_frame_image(i)
        mask, masked_img = mask_image(img, boxes)
        if save_path is not None:
            cv2.imwrite(os.path.join(save_path, '{:06d}.jpg'.format(i)),
                        masked_img, [cv2.IMWRITE_JPEG_QUALITY, 100])


def mask_video_ffmpeg(video, w_delta_percent, h_delta_percent, save_path):
    """Change the pixels of the input video outside boxes into black.

    Args
        video(video object): input video
        w_delta_percent(float): percentage of change in width
        h_delta_percent(float): percentage of change in width
        save_path(string): image save path

    """
    processes = []
    cmds = []
    for i in range(video.start_frame_index, video.end_frame_index + 1):
        boxes = video.get_frame_detection(i)
        boxes = resize_bboxes(boxes, w_delta_percent,
                              h_delta_percent, video.resolution)
        img_path = video.get_frame_image_name(i)
        img_name = os.path.basename(img_path)
        out_img_path = os.path.join(save_path, img_name)
        cmds.append(mask_image_ffmpeg_cmd(img_path, boxes, video.resolution,
                                          out_img_path))

    max_task = cpu_count()*10  # the number of cores in the system
    while True:
        print('{} jobs to do...'.format(len(cmds)))
        while cmds and len(processes) < max_task:
            task = cmds.pop()
            processes.append(Popen(task, stdin=open(os.devnull)))

        for p in processes:
            if p.poll() is not None:
                if p.returncode == 0:
                    processes.remove(p)
                else:
                    print(p.args, 'failed!')
                    import pdb
                    pdb.set_trace()
                    # sys.exit(1)

        if not processes and not cmds:
            break
        # else:
        #     time.sleep(0.01)

    print('Finished all image masking...', flush=True)


def mask_image_ffmpeg_cmd(input_img, boxes, resolution, output_img):
    """Return ffmpeg command that changes the image background into black.

    Args
        input_img(string): input image filename
        boxes(list): a list of boxes
                    box format [xmin, ymin, xmax, ymax, t, score, obj_id]
        resolution(tuple): (width, height)
        output_img(string): output image filename
    Return
        ffmpeg command(string)

    """
    filters = 'color=s={}x{}:c=black[bg];'.format(int(resolution[0]),
                                                  int(resolution[1]))
    if boxes:
        for i, box in enumerate(boxes):
            filters += '[0:v]crop=w={}:h={}:x={}:y={}[crop{}];'.format(
                box[2]-box[0], box[3]-box[1], box[0], box[1], i)
        for i, box in enumerate(boxes):
            if i == 0:
                filters += '[bg][crop{}]overlay=x={}:y={}:shortest=1[out{}];'\
                    .format(i, box[0], box[1], i)
            else:
                filters += '[out{}][crop{}]overlay=x={}:y={}[out{}];'.format(
                    i-1, i, box[0], box[1], i)

        filters = filters[:-1]
        return ['ffmpeg', '-loglevel', 'quiet', '-i', input_img, '-y',
                '-filter_complex', filters, '-map',
                '[out{}]'.format(len(boxes) - 1), "-qscale:v",
                "2.0", output_img, '-hide_banner']
    else:
        return ['ffmpeg', '-loglevel', 'quiet', '-i', input_img, '-y',
                '-filter_complex',
                filters+'[0:v][bg]overlay=x=0:y=0:shortest=1[out]', '-map',
                '[out]', "-qscale:v", "2.0", output_img, '-hide_banner']


def cpu_count():
    """Return the number of CPUs in the system."""
    num = 1
    if sys.platform == 'win32':
        try:
            num = int(os.environ['NUMBER_OF_PROCESSORS'])
        except (ValueError, KeyError):
            pass
    elif sys.platform == 'darwin':
        try:
            num = int(os.popen('sysctl -n hw.ncpu').read())
        except ValueError:
            pass
    else:
        try:
            num = os.sysconf('SC_NPROCESSORS_ONLN')
        except (ValueError, OSError, AttributeError):
            pass

    return num


def resize_bboxes(bboxes, w_delta_percent, h_delta_percent, resolution):
    """Resize bounding boxes.

    Args
        bboxes(list): a list of boxes.
                      [xmin, ymin, xmax, ymax, t, score, obj_id]
        w_delta_percent(float): percent of change over w
        h_delta_percent(float): percent of change over h
        resolution(tuple/list): (width, height)

    Return
        a list of resized bboxes

    """
    ret_bboxes = [resize_bbox(bbox, w_delta_percent, h_delta_percent,
                              resolution) for bbox in bboxes]
    return ret_bboxes


def resize_bbox(bbox, w_delta_percent, h_delta_percent, resolution):
    """Resize a bounding box.

    Args
        bbox(list): format is [xmin, ymin, xmax, ymax, t, score, obj_id]
        w_delta_percent(float): change percentage in width
        h_delta_percent(float): change percentage in height
        resolution(tuple/list): (width, height)

    """
    ret_bbox = bbox.copy()
    xmin, ymin, xmax, ymax = bbox[:4]
    w_delta = int(0.5 * (xmax - xmin) * w_delta_percent)
    h_delta = int(0.5 * (ymax - ymin) * h_delta_percent)
    ret_bbox[0] = max(xmin-w_delta, 0)
    ret_bbox[1] = max(ymin-h_delta, 0)
    ret_bbox[2] = min(xmax+w_delta, resolution[0])
    ret_bbox[3] = min(ymax+h_delta, resolution[1])
    return ret_bbox
