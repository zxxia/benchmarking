import cv2
from Vigil.helpers import remove_background, compute_video_size, \
    overlap_percentage, convert_box_coordinate
from utils.utils import load_metadata, compute_f1, create_dir
from utils.model_utils import load_full_model_detection, eval_single_image
import numpy as np
import pdb
import argparse
import json

# PATH = '/mnt/data/zhujun/dataset/Youtube/'
PATH = '/data/zxxia/videos/'

DATASET_LIST = sorted(['traffic', 'jp_hw', 'russia', 'tw_road',
                       'tw_under_bridge',  'nyc', 'lane_split', 'tw', 'tw1',
                       'russia1', 'park', 'drift', 'crossroad3', 'crossroad2',
                       'crossroad', 'driving2', 'crossroad4', 'driving1',
                       'driving_downtown', 'highway', 'highway_normal_traffic',
                       'jp', 'motorway'])

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


SHORT_VIDEO_LENGTH = 30  # seconds


def is_in_detections(detections, new_box):
    for box in detections:
        if box == new_box:
            # print('duplicate {} in {}'.format(new_box, detections))
            return True
    return False


def change_box_format(boxes):
    '''
    change box format from [x, y, w, h] to [xmin, ymin, xmax, ymax]
    '''
    return [convert_box_coordinate(box) for box in boxes]


def haar_detection(img, car_cascade, dataset):
    '''
    img: opencv image
    car_cascade
    return list of bounding boxes. box format: [xmin, ymin, xmax, ymax]
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = cv2.equalizeHist(gray)

    # cars = car_cascade.detectMultiScale(gray, 1.1, 1)
    cars = car_cascade.detectMultiScale(gray, SCALE_FACTOR_DICT[dataset],
                                            MIN_NEIGHBORS_DICT[dataset])

    cars = change_box_format(cars)
    return cars


def main():
    parser = argparse.ArgumentParser(description="vigil")
    parser.add_argument("--video", type=str, help="video name")

    args = parser.parse_args()
    dataset = args.video
    print(dataset)
    cascade_src_0 = 'haar_models/cars.xml'
    # cascade_src_0 = 'haar_models/cascade_data/cascade2.xml'
    # cascade_src_1 = 'haar_models/cars1.xml'
    # cascade_src_2 = 'haar_models/cars3.xml'
    # cascade_src_3 = 'haar_models/cars4.xml'
    # cascade_src_4 = 'haar_models/checkcas.xml'
    car_cascade_0 = cv2.CascadeClassifier(cascade_src_0)
    # car_cascade_1 = cv2.CascadeClassifier(cascade_src_1)
    # car_cascade_2 = cv2.CascadeClassifier(cascade_src_2)
    # car_cascade_3 = cv2.CascadeClassifier(cascade_src_3)
    # car_cascade_4 = cv2.CascadeClassifier(cascade_src_4)

    f_haar_dt = open('overfitting_results/haar_detections_{}.json'
                     .format(dataset), 'w', 1)
    haar_dets = {}

    with open('overfitting_results/haar_overfitting_{}.csv'.format(dataset),
              'w', 1) as f:
        f.write('video, f1, bw, scale factor, min neighbors, avg upload area, avg total obj area\n')
        resol = '720p/'
        img_path = PATH + dataset + '/' + resol
        # metadata_file = PATH + dataset + '/metadata.json'
        # print(metadata_file)
        metadata = load_metadata(PATH + dataset + '/metadata.json')
        #  frame_cnt = metadata['frame count']
        fps = metadata['frame rate']
        # resolution = metadata['resolution']
        resolution = [1280, 720]
        # gt_file = PATH + dataset + '/' + resol + \
        #     'profile/updated_gt_FasterRCNN_COCO.csv'
        gt_file = PATH + dataset + '/data/zxxia/benchmarking/results/' + resol + \
            'profile/updated_gt_FasterRCNN_COCO_no_filter.csv'
        gt, nb_frames = load_full_model_detection(gt_file)

        num_of_short_videos = nb_frames//(SHORT_VIDEO_LENGTH*fps)
        for i in range(num_of_short_videos):
            clip = dataset + '_' + str(i)
            tmp_folder = 'tmp/{}/'.format(clip)
            create_dir(tmp_folder)
            start_frame = i * SHORT_VIDEO_LENGTH * fps + 1
            end_frame = (i+1) * SHORT_VIDEO_LENGTH * fps
            tp = {}
            fp = {}
            fn = {}
            relative_up_areas = []
            tot_obj_areas = []
            for img_idx in range(start_frame, end_frame + 1):
                img_name = '{:06d}.jpg'.format(img_idx)
                img = cv2. imread(img_path+img_name)

                # get simple proposed regions which might have objects
                simple_model_dets = haar_detection(img, car_cascade_0, dataset)
                haar_dets[i] = simple_model_dets

                dt_boxes = list()
                gt_boxes = gt[img_idx]

                mask = np.zeros(img.shape, dtype=np.uint8)
                for box in simple_model_dets:
                    xmin, ymin, xmax, ymax = box[:4]
                    mask[ymin:ymax, xmin:xmax] = 1
                #  mask[mask<3] = 0
                #  mask[mask>=3] = 1

                processed_img = img.copy()
                processed_img *= mask

                #  processed_img, mask = remove_background(img, cars_0)
                # Uploaded area in terms of pixel count
                up_area = np.sum(mask)/3
                # relative uploaded area
                relative_up_area = up_area/(resolution[0]*resolution[1])
                relative_up_areas.append(relative_up_area)

                cv2.imwrite('{}/{:06d}.jpg'.format(tmp_folder, img_idx),
                            processed_img)

                tot_obj_area = 0
                for gt_box in gt_boxes:
                    xmin, ymin, xmax, ymax = gt_box[:4]
                    tot_obj_area += (xmax-xmin)*(ymax-ymin)/(resolution[0]*resolution[1])
                    ideal_pix_cnt = np.sum(np.ones((ymax-ymin, xmax-xmin, 3)))
                    actual_pix_cnt = np.sum(mask[ymin:ymax, xmin:xmax])
                    if actual_pix_cnt/ideal_pix_cnt > PIXEL_THRESHOLD:
                        dt_boxes.append(gt_box)
                tot_obj_areas.append(tot_obj_area)

                tp[img_idx], fp[img_idx], fn[img_idx] = \
                    eval_single_image(gt_boxes, dt_boxes)

                # for box in cars_0:
                #     x, y, xmax, ymax = box[:4]
                #     cv2.rectangle(gray, (x, y), (xmax, ymax),
                #                   (0, 255, 255), 2)

                #  for box in cars_1:
                    #  x, y, xmax, ymax = box[:4]
                    #  cv2.rectangle(img, (x, y), (xmax, ymax),
                                  #  (255, 255, 255), 2)
            #      for box in gt_boxes:
            #          x, y, xmax, ymax = box[:4]
            #          cv2.rectangle(img, (x, y), (xmax, ymax),
            #                        (0, 255, 0), 2)

            #      for box in dt_boxes:
            #          x, y, xmax, ymax = box[:4]
            #          cv2.rectangle(img, (x, y), (xmax, ymax),
            #                        (255, 0, 0), 2)
                # cv2.imshow('video', gray)
                # if cv2.waitKey(0) & 0xFF == ord('q'):
                #     break

            # cv2.destroyAllWindows()
            tp_total = sum(tp.values())
            fp_total = sum(fp.values())
            fn_total = sum(fn.values())
            f1 = compute_f1(tp_total, fp_total, fn_total)

            original_bw = compute_video_size(clip, img_path, start_frame,
                                             end_frame, fps, fps,
                                             resolution)
            bw = compute_video_size(clip, tmp_folder, start_frame, end_frame,
                                    fps, fps, resolution)

            print('{}_{}, start={}, end={}, f1={}, bw={}'
                  .format(dataset, i, start_frame, end_frame, f1,
                          bw/original_bw))
            f.write(','.join([clip, str(f1), str(bw/original_bw),
                              str(SCALE_FACTOR_DICT[dataset]),
                              str(MIN_NEIGHBORS_DICT[dataset]),
                              str(np.mean(relative_up_areas)),
                              str(np.mean(tot_obj_areas))+'\n']))
    json.dump(haar_dets, f_haar_dt, sorte_keys=True, indent=2)
    f_haar_dt.close()

if __name__ == '__main__':
    main()
