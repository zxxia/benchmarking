""" vigil overfitting scirpt  """
import argparse
# import pdb
import os
import cv2
from collections import defaultdict
import numpy as np
from Vigil.helpers import compute_video_size, convert_box_coordinate
from utils.utils import load_metadata, compute_f1, create_dir
from utils.model_utils import load_full_model_detection, eval_single_image, \
    filter_video_detections

PATH = '/mnt/data/zhujun/dataset/Youtube/'
# PATH = '/data/zxxia/videos/'

CAMERA_TYPES = {
        'static': ['crossroad', 'crossroad2', 'crossroad3',
                   'crossroad4', 'drift', 'highway', 'highway_normal_traffic',
                   'jp', 'jp_hw', 'motorway', 'nyc', 'russia',
                   'russia1', 'traffic', 'tw', 'tw1', 'tw_road',
                   'tw_under_bridge'],
        'moving': ['driving1', 'driving2', 'driving_downtown', 'park',
                   'lane_split']
}

SCALE_FACTOR = 1.2
MIN_NEIGHBORS = 10
PIXEL_THRESHOLD = 0.5

SCALE_FACTOR_DICT = {
    'traffic': 1.01,
    'jp_hw': 1.01,
    'russia': 1.01,
    'tw_road': 1.01,
    'tw_under_bridge': 1.01,
    'nyc': 1.01,
    'lane_split': 1.01,
    'tw': 1.01,
    'tw1': 1.01,
    'russia1': 1.01,
    'park': 1.01,
    'drift': 1.01,
    'crossroad3': 1.01,
    'crossroad2': 1.01,
    'crossroad': 1.01,
    'driving2': 1.01,
    'crossroad4': 1.01,
    'driving1': 1.01,
    'driving_downtown': 1.01,
    'highway': 1.01,
    'highway_normal_traffic': 1.01,
    'jp': 1.03,
    'motorway': 1.01,
}

MIN_NEIGHBORS_DICT = {
    'traffic': 1,
    'jp_hw': 1,
    'russia': 1,
    'tw_road': 1,
    'tw_under_bridge': 1,
    'nyc': 1,
    'lane_split': 1,
    'tw': 1,
    'tw1': 1,
    'russia1': 1,
    'park': 1,
    'drift': 1,
    'crossroad3': 1,
    'crossroad2': 1,
    'crossroad': 1,
    'driving2': 1,
    'crossroad4': 1,
    'driving1': 1,
    'driving_downtown': 1,
    'highway': 1,
    'highway_normal_traffic': 1,
    'jp': 1,
    'motorway': 1,
}


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
        # dets = json.load(f_det, object_pairs_hook=lambda pairs:
        #                  {int(k): v for k, v in pairs})
        # det = json.load(f_det)
    return dets


def haar_detection(img, car_cascade, dataset):
    """ img: opencv image
        car_cascade
        return list of bounding boxes. box format: [xmin, ymin, xmax, ymax]
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = cv2.equalizeHist(gray)

    # cars = car_cascade.detectMultiScale(gray, 1.1, 1)
    cars = car_cascade.detectMultiScale(gray, SCALE_FACTOR_DICT[dataset],
                                        MIN_NEIGHBORS_DICT[dataset])

    cars = change_box_format(cars)
    return cars


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
    """ Crop an image based on simple model detections"""
    mask = np.zeros(img.shape, dtype=np.uint8)
    for box in simple_model_dets:
        xmin, ymin, xmax, ymax = box[:4]
        mask[ymin:ymax, xmin:xmax] = 1

    processed_img = img.copy()
    processed_img *= mask
    cv2.imwrite('{}/{:06d}.jpg'.format(tmp_folder, img_idx), processed_img)
    return mask


def main():
    """ Vigil main logic """
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
    # parser.add_argument("--fps", type=int, default=0, help="frame rate")
    # parser.add_argument("--resolution", nargs='+', type=int,
    #                     default=[], action='store', help="video resolution")
    # parser.add_argument("--format", type=str, default='{:06d}.jpg',
    #                     help="image name format")
    # parser.add_argument("--profile_length", type=int,
    #                     help="profile length in seconds")
    # parser.add_argument("--target_f1", type=float, help="target F1 score")

    args = parser.parse_args()
    # path = args.path
    dataset = args.video
    output_file = args.output
    metadata_file = args.metadata
    # log_file = args.log
    short_video_length = args.short_video_length
    offset = args.offset
    # target_f1 = args.target_f1
    # profile_length = args.profile_length

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

    print(dataset)
    # cascade_src = 'haar_models/cars.xml'
    # car_cascade = cv2.CascadeClassifier(cascade_src)

    # f_haar_dt = open(log_file, 'w', 1)
    # haar_dets = {}
    # haar_dt_file = 'haar_detections_new/haar_{}.csv'.format(dataset)
    # haar_dets = load_haar_detection(haar_dt_file)
    # haar_dets = filter_haar_detection(haar_dets, height_range=(720//20, 720))
    with open(output_file, 'w', 1) as f_out:
        f_out.write('video, f1, bw, scale factor, min neighbors, '
                    'avg upload area, avg total obj area\n')
        # metadata_file = path + dataset + '/metadata.json'
        # print(metadata_file)
        # gt_file = path + dataset + '/' + resol + \
        #     'profile/updated_gt_FasterRCNN_COCO.csv'
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
        num_of_short_videos = nb_frames//(short_video_length*fps)
        for i in range(num_of_short_videos):
            clip = dataset + '_' + str(i)
            tmp_folder = 'tmp/{}/'.format(clip)
            create_dir(tmp_folder)
            start_frame = i * short_video_length * fps + 1 + offset * fps
            end_frame = (i+1) * short_video_length * fps + offset * fps
            tpos = {}
            fpos = {}
            fneg = {}
            relative_up_areas = []
            tot_obj_areas = []
            for img_idx in range(start_frame, end_frame+1):
                img_name = '{:06d}.jpg'.format(img_idx)
                img = cv2. imread(img_path+img_name)

                # get simple proposed regions which might have objects
                # simple_model_dets = haar_detection(img, car_cascade, dataset)

                # TODO: replace haar detections with mobilenet ssd detections
                # haar_dets[i] = simple_model_dets
                # simple_model_dets = haar_dets[img_idx]
                dt_boxes = list()
                gt_boxes = gtruth[img_idx]

                mask = crop_image(img, img_idx, gt_boxes, tmp_folder)

                # Uploaded area in terms of pixel count
                up_area = np.sum(mask)/3
                # relative uploaded area
                relative_up_area = up_area/(resolution[0]*resolution[1])
                relative_up_areas.append(relative_up_area)

                tot_obj_area = 0
                for gt_box in gt_boxes:
                    xmin, ymin, xmax, ymax = gt_box[:4]
                    tot_obj_area += (xmax-xmin)*(ymax-ymin) /\
                        (resolution[0]*resolution[1])
                    ideal_pix_cnt = np.sum(np.ones((ymax-ymin, xmax-xmin, 3)))
                    actual_pix_cnt = np.sum(mask[ymin:ymax, xmin:xmax])
                    if actual_pix_cnt/ideal_pix_cnt > PIXEL_THRESHOLD:
                        dt_boxes.append(gt_box)
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
            for img_idx in range(start_frame, end_frame+1):
                os.remove('{}/{:06d}.jpg'.format(tmp_folder, img_idx))

            print('{}_{}, start={}, end={}, f1={}, bw={}'
                  .format(dataset, i, start_frame, end_frame, f1_score,
                          bandwidth/original_bw))
            f_out.write(','.join([clip, str(f1_score),
                                  str(bandwidth/original_bw),
                                  str(SCALE_FACTOR_DICT[dataset]),
                                  str(MIN_NEIGHBORS_DICT[dataset]),
                                  str(np.mean(relative_up_areas)),
                                  str(np.mean(tot_obj_areas))+'\n']))


if __name__ == '__main__':
    main()
