"""Vigil Implementation."""
import time
import os
import subprocess
import cv2
import numpy as np
from benchmarking.utils.model_utils import eval_single_image
from benchmarking.utils.utils import compute_f1


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
        for i in range(frame_range[0], frame_range[1] + 1):

            # relative uploaded area
            # relative_up_area = 0
            # simple_dt_boxes = mobilenet_dt[img_idx]
            # for simple_dt_box in simple_dt_boxes:
            #     simple_dt_box = resize_bbox(simple_dt_box, 0.2, 0.2,
            #                                 resolution)
            #     xmin, ymin, xmax, ymax = simple_dt_box[:4]
            #     relative_up_area += (xmax-xmin)*(ymax-ymin) /\
            #         (resolution[0]*resolution[1])
            # relative_up_areas.append(relative_up_area)

            # tot_obj_area = 0
            # for gt_box in gt_boxes:
            #     xmin, ymin, xmax, ymax = gt_box[:4]
            #     tot_obj_area += (xmax-xmin)*(ymax-ymin) /\
            #         (resolution[0]*resolution[1])
            # tot_obj_areas.append(tot_obj_area)

            tpos[i], fpos[i], fneg[i] = eval_single_image(
                original_video.get_frame_detection(i),
                video.get_frame_detection(i))
        tp_total = sum(tpos.values())
        fp_total = sum(fpos.values())
        fn_total = sum(fneg.values())
        f1_score = compute_f1(tp_total, fp_total, fn_total)
        # TODO: change video name
        original_bw = original_video.encode(
            os.path.join(video_save_path, video_name + '.mp4'),
            list(range(frame_range[0], frame_range[1] + 1)),
            original_video.frame_rate)
        bw = video.encode(os.path.join(video_save_path, video_name+'.mp4'),
                          list(range(frame_range[0], frame_range[1] + 1)),
                          video.frame_rate)
        return bw/original_bw, f1_score


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
    """Change the pixels of the input video outside boxes into black."""
    processes = []
    for i in range(video.start_frame_index, video.end_frame_index + 1):
        boxes = video.get_frame_detection(i)
        boxes = resize_bboxes(boxes, w_delta_percent,
                              h_delta_percent, video.resolution)
        img_path = video.get_frame_image_name(i)
        img_name = os.path.basename(img_path)
        out_img_path = os.path.join(save_path, img_name)
        processes.append(mask_image_ffmpeg(img_path, boxes, out_img_path))

    while processes:
        # remove finished processes from the list (O(N**2))
        print('{} tasks to run...'.format(len(processes)), flush=True)
        for p in processes[:]:
            if p.poll() is not None:  # process ended
                # print(p.stdout.read(), end='')  # read the rest
                # p.stdout.close()
                processes.remove(p)
        time.sleep(2)
    print('Finished all image masking...', flush=True)


def mask_image_ffmpeg(input_img, boxes, output_img):
    """Change the pixels of the input image outside boxes into black.

    Args
        input_img(string): input image filename
        boxes(list): a list of boxes
                    box format [xmin, ymin, xmax, ymax, t, score, obj_id]
        output_img(string): output image filename

    """
    filters = 'color=s=1280x720:c=black[bg];'
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
    cmd = ['ffmpeg', '-loglevel', 'quiet', '-i', input_img, '-y',
           '-filter_complex', filters, '-map',
           '[out{}]'.format(len(boxes) - 1), "-qscale:v",
           "2.0", output_img, '-hide_banner']
    # print(cmd)
    # subprocess.run(cmd, check=True)

    # return subprocess.Popen(cmd, stdin=subprocess.PIPE)
    # TODO: solve the terminal recovery issue
    return subprocess.Popen(cmd)


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


def load_vigil_results(filename):
    """Load vigil result file."""
    videos = []
    bw_list = []
    acc_list = []
    with open(filename, 'r') as f:
        f.readline()
        for line in f:
            cols = line.strip().split(',')
            videos.append(cols[0])
            bw_list.append(float(cols[1]))
            acc_list.append(float(cols[2]))

    return videos, bw_list, acc_list
