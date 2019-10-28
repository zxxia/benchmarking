""" vigil overfitting scirpt  """
import argparse
# import pdb
# import os
from collections import defaultdict
import cv2
import numpy as np
from Vigil.helpers import compute_video_size, convert_box_coordinate
from utils.utils import load_metadata, compute_f1
from utils.model_utils import load_full_model_detection, eval_single_image, \
    filter_video_detections
from constants import CAMERA_TYPES

PATH = '/data/zxxia/videos/'


SCALE_FACTOR = 1.2
MIN_NEIGHBORS = 10
PIXEL_THRESHOLD = 0.5


def change_box_format(boxes):
    """ Change box format from [x, y, w, h] to [xmin, ymin, xmax, ymax] """
    return [convert_box_coordinate(box) for box in boxes]


def load_haar_detection(filename):
    """ Load haar detection (csv) """
    dets = defaultdict(list)
    with open(filename, 'r') as f_det:
        for line in f_det:
            cols = line.strip().split(',')
            frame_idx = int(cols[0])
            xmin = int(cols[1])
            ymin = int(cols[2])
            xmax = int(cols[3])
            ymax = int(cols[4])
            dets[frame_idx].append([xmin, ymin, xmax, ymax])
    return dets


# def haar_detection(img, car_cascade, dataset):
#     """ img: opencv image
#         car_cascade
#         return list of bounding boxes. box format: [xmin, ymin, xmax, ymax]
#     """
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # gray = cv2.equalizeHist(gray)
#
#     # cars = car_cascade.detectMultiScale(gray, 1.1, 1)
#     cars = car_cascade.detectMultiScale(gray, SCALE_FACTOR_DICT[dataset],
#                                         MIN_NEIGHBORS_DICT[dataset])
#
#     cars = change_box_format(cars)
#     return cars


def filter_haar_detection(dets, width_range=None, height_range=None):
    """ filter haar detections
    dets: a dict mapping frame index to a list of object bounding boxes.
          box format [xmin, ymin, xmax, ymax]"""
    assert (width_range is None) or \
           (isinstance(width_range, tuple) and len(width_range) == 2 and
            width_range[0] <= width_range[1]), \
        'width range needs to be a length 2 tuple or None'
    assert (height_range is None) or \
           (isinstance(height_range, tuple) and len(height_range) == 2 and
            height_range[0] <= height_range[1]), \
        'height range needs to be a length 2 tuple or None'
    filtered_dets = dict()
    for frame_idx in dets:
        filtered_boxes = list()
        for box in dets[frame_idx]:
            xmin, ymin, xmax, ymax = box
            width = xmax - xmin
            height = ymax - ymin
            if width_range is not None and \
               (width < width_range[0] or width > width_range[1]):
                continue
            if height_range is not None and \
               (height < height_range[0] or height > height_range[1]):
                continue
            filtered_boxes.append(box)
        filtered_dets[frame_idx] = filtered_boxes
    return filtered_dets


def visualize(img, bboxes):
    """ visualize a frame """
    for box in bboxes:
        xmin, ymin, xmax, ymax = box[:4]
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)
    cv2.imshow('video', img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        return False
    return True


def crop_image(img, img_idx, simple_model_dets, tmp_folder):
    """ Crop images based on simple model detections"""
    mask = np.zeros(img.shape, dtype=np.uint8)
    for box in simple_model_dets:
        xmin, ymin, xmax, ymax = box[:4]
        mask[ymin:ymax, xmin:xmax] = 1

    processed_img = img.copy()
    processed_img *= mask
    cv2.imwrite('{}/{:06d}.jpg'.format(tmp_folder, img_idx), processed_img)
    return mask


def resize_bbox(bbox, w_delta_percent, h_delta_percent, resolution):
    """Resize a bounding box.
       box format is [xmin, ymin, xmax, ymax, t, score, obj_id]"""
    ret_bbox = bbox.copy()
    xmin, ymin, xmax, ymax = bbox[:4]
    w_delta = int(0.5 * (xmax - xmin) * w_delta_percent)
    h_delta = int(0.5 * (ymax - ymin) * h_delta_percent)
    ret_bbox[0] = max(xmin-w_delta, 0)
    ret_bbox[1] = max(ymin-h_delta, 0)
    ret_bbox[2] = min(xmax+w_delta, resolution[0])
    ret_bbox[3] = min(ymax+h_delta, resolution[1])
    return ret_bbox


def resize_bboxes(bboxes, w_delta_percent, h_delta_percent, resolution):
    """Resize a bounding box.
       box format is [xmin, ymin, xmax, ymax, t, score, obj_id]"""
    ret_bboxes = [resize_bbox(bbox, w_delta_percent, h_delta_percent,
                              resolution) for bbox in bboxes]
    return ret_bboxes


def remove_background(img_path, start_frame, end_frame, simple_model_dets,
                      output_folder, resolution):
    """ Remove background by setting it to black color and save the processed
        images into output folder
        return a list of uploaded area in terms of pixel count """
    relative_up_areas = list()
    for img_idx in range(start_frame, end_frame+1):
        img_name = '{:06d}.jpg'.format(img_idx)
        img = cv2. imread(img_path+img_name)
        simple_model_frame_dt = simple_model_dets[img_idx]
        # increase the width and the height by certain percent
        resized_simple_model_frame_dt = resize_bboxes(simple_model_frame_dt,
                                                      0.2, 0.2, resolution)

        mask = crop_image(img, img_idx, resized_simple_model_frame_dt,
                          output_folder)
        # Uploaded area in terms of pixel count
        up_area = np.sum(mask)/3
        # relative uploaded area
        relative_up_area = up_area/(resolution[0]*resolution[1])
        relative_up_areas.append(relative_up_area)

        if (img_idx - start_frame) % 100 == 0:
            print("Removed background for {} images..."
                  .format((img_idx-start_frame)))
    return relative_up_areas


def parse_args():
    """ parse input arguments """
    parser = argparse.ArgumentParser(description="vigil")
    parser.add_argument("--path", type=str, help="path contains all datasets")
    parser.add_argument("--video", type=str, help="video name")
    parser.add_argument("--metadata", type=str, default='',
                        help="metadata file in Json")
    parser.add_argument("--output", type=str, help="output result file")
    # parser.add_argument("--log", type=str, help="log middle file")
    parser.add_argument("--short_video_length", type=int,
                        help="short video length in seconds")
    parser.add_argument("--offset", type=int,
                        help="offset from beginning of the video in seconds")
    parser.add_argument("--fps", type=int, default=0, help="frame rate")
    parser.add_argument("--resolution", nargs='+', type=int,
                        default=[], action='store', help="video resolution")
    args = parser.parse_args()
    return args


def main():
    """ Vigil main logic """
    args = parse_args()
    # path = args.path
    dataset = args.video
    output_file = args.output
    metadata_file = args.metadata
    short_video_length = args.short_video_length
    offset = args.offset

    if metadata_file:
        metadata = load_metadata(metadata_file)
        resolution = metadata['resolution']
        # frame_cnt = metadata['frame count']
        fps = metadata['frame rate']
    else:
        fps = args.fps
        resolution = args.resolution
    resolution = [1280, 720]

    resol = '720p/'
    img_path = PATH + dataset + '/' + resol

    # Load haar detection results
    # haar_dt_file = 'haar_detections_new/haar_{}.csv'.format(dataset)
    # haar_dets = load_haar_detection(haar_dt_file)
    # haar_dets = filter_haar_detection(haar_dets, height_range=(720//20, 720))

    # Load mobilenet detection results
    mobilenet_file = '/data/zxxia/benchmarking/results/videos/'+dataset+'/'+resol + \
        'profile/updated_gt_mobilenet_COCO_no_filter.csv'
    mobilenet_dt, nb_frames = load_full_model_detection(mobilenet_file)
    if dataset in CAMERA_TYPES['static']:
        # mobilenet_dt = filter_video_detections(mobilenet_dt,
        #                                        width_range=(0, 1280/2),
        #                                        height_range=(720//20, 720/2),
        #                                        target_types={3, 8})
        mobilenet_dt = filter_video_detections(mobilenet_dt,
                                               width_range=(0, 1280/2),
                                               height_range=(0, 720/2))
    else:
        # mobilenet_dt = filter_video_detections(mobilenet_dt,
        #                                        height_range=(720//20, 720),
        #                                        target_types={3, 8})
        mobilenet_dt = filter_video_detections(mobilenet_dt)

    # for frame_idx, bboxes in mobilenet_dt.items():
    #     for box_pos, box in enumerate(bboxes):
    #         box[4] = 3
    #         bboxes[box_pos] = box
    #     mobilenet_dt[frame_idx] = bboxes

    # Remove background based on simple model results
    tmp_folder = '/data/zxxia/blackbg/{}/'.format(dataset)
    # create_dir(tmp_folder)
    # relative_areas = remove_background(img_path, 1, nb_frames, mobilenet_dt,
    #                                    tmp_folder, resolution)
    # return

    # Load fastercnn detections on blacked background images
    dt_file = '/data/zxxia/blackbg/'+dataset+'/' + \
        'profile/updated_gt_FasterRCNN_COCO_no_filter.csv'
    dets, nb_frames = load_full_model_detection(dt_file)
    if dataset in CAMERA_TYPES['static']:
        dets = filter_video_detections(dets,
                                       width_range=(0, 1280/2),
                                       height_range=(720//20, 720/2),
                                       target_types={3, 8})
    else:
        dets = filter_video_detections(dets,
                                       height_range=(720//20, 720),
                                       target_types={3, 8})
    for frame_idx, bboxes in dets.items():
        for box_pos, box in enumerate(bboxes):
            box[4] = 3
            bboxes[box_pos] = box
        dets[frame_idx] = bboxes

    # Load ground truth
    gt_file = '/data/zxxia/benchmarking/results/videos/'+dataset+'/'+resol + \
        'profile/updated_gt_FasterRCNN_COCO_no_filter.csv'
    gtruth, nb_frames = load_full_model_detection(gt_file)
    if dataset in CAMERA_TYPES['static']:
        gtruth = filter_video_detections(gtruth,
                                         width_range=(0, 1280/2),
                                         height_range=(720//20, 720/2),
                                         target_types={3, 8})
    else:
        gtruth = filter_video_detections(gtruth,
                                         height_range=(720//20, 720),
                                         target_types={3, 8})
    for frame_idx, bboxes in gtruth.items():
        for box_pos, box in enumerate(bboxes):
            box[4] = 3
            bboxes[box_pos] = box
        gtruth[frame_idx] = bboxes

    # do the video bandwidth computation
    with open(output_file, 'w', 1) as f_out:
        f_out.write('video, f1, bw, avg upload area, avg total obj area\n')
        num_of_short_videos = nb_frames//(short_video_length*fps)
        for i in range(num_of_short_videos):
            clip = dataset + '_' + str(i)
            # tmp_folder = '/data/zxxia/blackbg/{}/'.format(dataset)
            # create_dir(tmp_folder)
            start_frame = i * short_video_length * fps + 1 + offset * fps
            end_frame = (i+1) * short_video_length * fps + offset * fps
            tpos = {}
            fpos = {}
            fneg = {}
            relative_up_areas = []
            tot_obj_areas = []
            for img_idx in range(start_frame, end_frame + 1):
                dt_boxes = dets[img_idx]
                # dt_boxes = mobilenet_dt[img_idx]
                gt_boxes = gtruth[img_idx]

                # relative uploaded area
                relative_up_area = 0
                simple_dt_boxes = mobilenet_dt[img_idx]
                for simple_dt_box in simple_dt_boxes:
                    simple_dt_box = resize_bbox(simple_dt_box, 0.2, 0.2,
                                                resolution)
                    xmin, ymin, xmax, ymax = simple_dt_box[:4]
                    relative_up_area += (xmax-xmin)*(ymax-ymin) /\
                        (resolution[0]*resolution[1])
                relative_up_areas.append(relative_up_area)

                tot_obj_area = 0
                for gt_box in gt_boxes:
                    xmin, ymin, xmax, ymax = gt_box[:4]
                    tot_obj_area += (xmax-xmin)*(ymax-ymin) /\
                        (resolution[0]*resolution[1])
                tot_obj_areas.append(tot_obj_area)

                tpos[img_idx], fpos[img_idx], fneg[img_idx] = \
                    eval_single_image(gt_boxes, dt_boxes)

            tp_total = sum(tpos.values())
            fp_total = sum(fpos.values())
            fn_total = sum(fneg.values())
            f1_score = compute_f1(tp_total, fp_total, fn_total)

            original_bw = compute_video_size(clip, img_path, start_frame,
                                             end_frame, fps, fps, resolution)
            bandwidth = compute_video_size(clip, tmp_folder, start_frame,
                                           end_frame, fps, fps, resolution)

            print('{}_{}, start={}, end={}, f1={}, bw={}'
                  .format(dataset, i, start_frame, end_frame, f1_score,
                          bandwidth/original_bw))
            f_out.write(','.join([clip, str(f1_score),
                                  str(bandwidth/original_bw),
                                  str(np.mean(relative_up_areas)),
                                  str(np.mean(tot_obj_areas))+'\n']))


if __name__ == '__main__':
    main()
