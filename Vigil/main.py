import cv2
from Vigil.helpers import remove_background, compute_video_size, overlap_percentage, convert_box_coordinate
from utils.utils import load_metadata, compute_f1
from utils.model_utils import load_full_model_detection, eval_single_image
import numpy as np
import pdb

PATH = '/mnt/data/zhujun/dataset/Youtube/'

DATASET_LIST = ['highway']#, 'jp', 'motorway']
#  sorted(['traffic', 'jp_hw', 'russia', 'tw_road',
#          'tw_under_bridge',  'nyc',
#          'lane_split', 'tw', 'tw1',  'russia1', 'park',
#          'drift', 'crossroad3','crossroad2', 'crossroad', #'driving2','crossroad4', 'driving1','driving_downtown','highway','highway_normal_traffic','jp',
#          'motorway'])

SCALE_FACTOR = 1.01
MIN_NEIGHBORS = 1
PIXEL_THRESHOLD = 0.4


TARGET_DATASET = ['crossroad2_11', 'crossroad2_12', 'crossroad2_13',
                  'crossroad2_14', 'crossroad2_18', 'crossroad2_2',
                  'crossroad2_20', 'crossroad2_21', 'crossroad2_22',
                  'crossroad2_23', 'crossroad2_25', 'crossroad2_26',
                  'crossroad2_30', 'crossroad2_34', 'crossroad2_35',
                  'crossroad2_36', 'crossroad2_37', 'crossroad2_39',
                  'crossroad2_48', 'crossroad2_49', 'crossroad2_50',
                  'crossroad2_52', 'crossroad2_56', 'crossroad2_57',
                  'crossroad2_6', 'crossroad2_61', 'crossroad2_62',
                  'crossroad2_64', 'crossroad2_69', 'crossroad2_7',
                  'crossroad2_70', 'crossroad2_73', 'crossroad2_74',
                  'crossroad2_77', 'crossroad2_78', 'crossroad2_8',
                  'crossroad2_83', 'crossroad2_86', 'crossroad2_87',
                  'crossroad2_89', 'crossroad2_90', 'crossroad2_93',
                  'crossroad3_10', 'crossroad3_16', 'crossroad3_35',
                  'crossroad3_5', 'crossroad3_7', 'crossroad4_12',
                  'crossroad4_13', 'crossroad4_14', 'crossroad4_22',
                  'crossroad4_31', 'crossroad4_37', 'crossroad4_6',
                  'crossroad4_7', 'crossroad4_8', 'crossroad_1',
                  'crossroad_11', 'crossroad_15', 'crossroad_16',
                  'crossroad_21', 'crossroad_22', 'crossroad_25',
                  'crossroad_28', 'crossroad_31', 'crossroad_33',
                  'crossroad_34', 'crossroad_35', 'crossroad_36',
                  'crossroad_39', 'crossroad_4', 'crossroad_40',
                  'crossroad_41', 'crossroad_47', 'crossroad_48',
                  'crossroad_5', 'crossroad_62', 'crossroad_63', 'crossroad_7',
                  'drift_0', 'drift_1', 'drift_16', 'drift_17', 'drift_21',
                  'drift_27', 'drift_32', 'drift_34', 'drift_36',
                  'driving1_25', 'driving_downtown_13', 'driving_downtown_15',
                  'driving_downtown_2', 'driving_downtown_3',
                  'driving_downtown_45', 'driving_downtown_49',
                  'driving_downtown_51', 'driving_downtown_52',
                  'driving_downtown_66',
                  'highway_0', 'highway_1',
                  'highway_10', 'highway_11', 'highway_12', 'highway_13',
                  'highway_18', 'highway_2', 'highway_29', 'highway_3',
                  'highway_30', 'highway_31', 'highway_32', 'highway_33',
                  'highway_34', 'highway_35', 'highway_36', 'highway_37',
                  'highway_39', 'highway_4', 'highway_43', 'highway_44',
                  'highway_45', 'highway_46', 'highway_47', 'highway_48',
                  'highway_5', 'highway_52', 'highway_55', 'highway_56',
                  'highway_57', 'highway_58', 'highway_59', 'highway_6',
                  'highway_60', 'highway_61', 'highway_62', 'highway_63',
                  'highway_64', 'highway_65', 'highway_67', 'highway_9',
                  'jp_0', 'jp_1', 'jp_10', 'jp_11', 'jp_14', 'jp_2', 'jp_22',
                  'jp_24', 'jp_26', 'jp_27', 'jp_28', 'jp_30', 'jp_31',
                  'jp_32', 'jp_33', 'jp_34', 'jp_35', 'jp_38', 'jp_4', 'jp_5',
                  'jp_6',
                  'lane_split_7', 'lane_split_8',
                  'motorway_0',
                  'motorway_1', 'motorway_2', 'motorway_3', 'motorway_4',
                  'motorway_5', 'motorway_6',
                  'nyc_15', 'nyc_29',
                  'park_15',
                  'park_17', 'park_19', 'park_21', 'park_27', 'park_29',
                  'park_5', 'park_7',
                  'russia1_11', 'russia1_17', 'russia1_23',
                  'russia1_27', 'russia1_38', 'russia1_8', 'russia_0',
                  'russia_2', 'russia_7', 'traffic_1', 'tw1_12', 'tw1_18',
                  'tw1_20', 'tw1_23', 'tw1_24', 'tw1_27', 'tw1_30', 'tw1_36',
                  'tw1_37', 'tw_21', 'tw_3', 'tw_33', 'tw_9',
                  'tw_under_bridge_0']


SHORT_VIDEO_LENGTH = 30  # seconds


def is_in_detections(detections, new_box):
    for box in detections:
        if box == new_box:
            # print('duplicate {} in {}'.format(new_box, detections))
            return True
    return False


def change_box_format(boxes):
    return [convert_box_coordinate(box) for box in boxes]


def main():
    cascade_src_0 = 'haar_models/cars.xml'
    cascade_src_1 = 'haar_models/cars1.xml'
    cascade_src_2 = 'haar_models/cars3.xml'
    cascade_src_3 = 'haar_models/cars4.xml'
    cascade_src_4 = 'haar_models/checkcas.xml'
    car_cascade_0 = cv2.CascadeClassifier(cascade_src_0)
    car_cascade_1 = cv2.CascadeClassifier(cascade_src_1)
    car_cascade_2 = cv2.CascadeClassifier(cascade_src_2)
    car_cascade_3 = cv2.CascadeClassifier(cascade_src_3)
    car_cascade_4 = cv2.CascadeClassifier(cascade_src_4)

    with open('haar_test.csv', 'w') as f:
        f.write('video, f1, bw, scale factor, min neighbors, avg upload area, avg total obj area\n')
        for dataset in DATASET_LIST:
            resol = '720p/'
            img_path = PATH + dataset + '/' + resol
            metadata = load_metadata(PATH + dataset + '/metadata.json')
            #  frame_cnt = metadata['frame count']
            fps = metadata['frame rate']
            resolution = metadata['resolution']
            resolution = [1280, 720]
            gt_file = PATH + dataset + '/' + resol + \
                'profile/updated_gt_FasterRCNN_COCO.csv'
            gt, nb_frames = load_full_model_detection(gt_file)

            num_of_short_videos = nb_frames//(SHORT_VIDEO_LENGTH*fps)
            for i in range(num_of_short_videos):
                if dataset + '_' + str(i) not in TARGET_DATASET:
                    continue
                start_frame = i * SHORT_VIDEO_LENGTH * fps + 1
                end_frame = (i+1) * SHORT_VIDEO_LENGTH * fps
                tp = {}
                fp = {}
                fn = {}
                relative_up_areas = []
                tot_obj_areas = []
                for img_idx in range(start_frame, end_frame + 1):
                    img_name = '{:06d}.jpg'.format(img_idx)
                    img = cv2. imread(img_path + img_name)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    # cars = car_cascade.detectMultiScale(gray, 1.1, 1)
                    cars_0 = car_cascade_0.detectMultiScale(gray, SCALE_FACTOR,
                                                            MIN_NEIGHBORS)
                    #  cars_1 = car_cascade_1.detectMultiScale(gray, 1.08, 3)
                    #  cars_2 = car_cascade_2.detectMultiScale(gray, 1.1, 5)
                    #  cars_3 = car_cascade_3.detectMultiScale(gray, 1.1, 5)
                    #  cars_4 = car_cascade_4.detectMultiScale(gray, 1.1, 5)
                    # , minSize=(30,30))
                    #  print('Finished HAAR Detection on {}_{} frame {}...'
                    #        .format(dataset, i, img_idx))

                    cars_0 = change_box_format(cars_0)
                    #  cars_1 = change_box_format(cars_1)
                    #  cars_2 = change_box_format(cars_2)
                    #  cars_3 = change_box_format(cars_3)
                    #  cars_4 = change_box_format(cars_4)

                    dt_boxes = list()
                    gt_boxes = gt[img_idx]

                    mask = np.zeros(img.shape, dtype=np.uint8)
                    for box in cars_0:
                        xmin, ymin, xmax, ymax = box[:4]
                        mask[ymin:ymax, xmin:xmax] = 1
                    #  for box in cars_1:
                        #  xmin, ymin, xmax, ymax = box[:4]
                        #  mask[ymin:ymax, xmin:xmax] = 1
                    #  for box in cars_2:
                        #  xmin, ymin, xmax, ymax = box[:4]
                        #  mask[ymin:ymax, xmin:xmax] += 1
                    #  for box in cars_3:
                        #  xmin, ymin, xmax, ymax = box[:4]
                        #  mask[ymin:ymax, xmin:xmax] += 1
                    #  for box in cars_4:
                        #  xmin, ymin, xmax, ymax = box[:4]
                        #  mask[ymin:ymax, xmin:xmax] += 1
                    #  mask[mask<3] = 0
                    #  mask[mask>=3] = 1

                    processed_img = img.copy()
                    processed_img *= mask

                    #  processed_img, mask = remove_background(img, cars_0)
                    # Uploaded area in terms of pixel count
                    up_area = np.sum(mask) / 3
                    # relative uploaded area
                    relative_up_area = up_area / (resolution[0] * resolution[1])
                    relative_up_areas.append(relative_up_area)

                    cv2.imwrite('tmp/{:06d}.jpg'.format(img_idx),
                                processed_img)

                    tot_obj_area = 0
                    for gt_box in gt_boxes:
                        xmin, ymin, xmax, ymax = gt_box[:4]
                        tot_obj_area += (xmax-xmin)*(ymax-ymin)/(resolution[0]*resolution[1])
                        ideal_pix_cnt = np.sum(np.ones((ymax-ymin,
                                                        xmax-xmin, 3)))
                        actual_pix_cnt = np.sum(mask[ymin:ymax, xmin:xmax])
                        if actual_pix_cnt/ideal_pix_cnt > PIXEL_THRESHOLD:
                            dt_boxes.append(gt_box)
                    tot_obj_areas.append(tot_obj_area)

                    tp[img_idx], fp[img_idx], fn[img_idx] = \
                        eval_single_image(gt_boxes, dt_boxes)

                    for box in cars_0:
                        x, y, xmax, ymax = box[:4]
                        cv2.rectangle(img, (x, y), (xmax, ymax),
                                      (0, 255, 255), 2)

                    #  for box in cars_1:
                        #  x, y, xmax, ymax = box[:4]
                        #  cv2.rectangle(img, (x, y), (xmax, ymax),
                                      #  (255, 255, 255), 2)
                    for box in gt_boxes:
                        x, y, xmax, ymax = box[:4]
                        cv2.rectangle(img, (x, y), (xmax, ymax),
                                      (0, 255, 0), 2)

                    for box in dt_boxes:
                        x, y, xmax, ymax = box[:4]
                        cv2.rectangle(img, (x, y), (xmax, ymax),
                                      (255, 0, 0), 2)
                    cv2.imshow('video', img)
                    if cv2.waitKey(0) == 27:
                        break

                cv2.destroyAllWindows()
                tp_total = sum(tp.values())
                fp_total = sum(fp.values())
                fn_total = sum(fn.values())
                f1 = compute_f1(tp_total, fp_total, fn_total)

                original_bw = compute_video_size(img_path, start_frame,
                                                 end_frame, fps, fps,
                                                 resolution)
                bw = compute_video_size('tmp/', start_frame, end_frame, fps,
                                        fps, resolution)

                print('{}_{}, start={}, end={}, f1={}, bw={}'
                      .format(dataset, i, start_frame, end_frame, f1,
                              bw/original_bw))
                f.write(','.join([dataset+'_'+str(i), str(f1),
                                  str(bw/original_bw), str(SCALE_FACTOR),
                                  str(MIN_NEIGHBORS),
                                  str(np.mean(relative_up_areas)),
                                  str(np.mean(tot_obj_areas))+'\n']))
                # break


if __name__ == '__main__':
    main()



# for dt_box in cars:
    # overlaps = list()
    # for gt_box in gt_boxes:
        # percent = overlap_percentage(dt_box, gt_box[:4])
        # overlaps.append(percent)
        # print('frame={}, percent={}, overlaps={}'.format(img_idx, percent, overlaps))
    # if overlaps:
        # max_overlap_idx = np.argmax(overlaps)
        # if not is_in_detections(dt_boxes, gt_boxes[max_overlap_idx]): # and overlaps[max_overlap_idx] > 0.5:
            # dt_boxes.append(gt_boxes[max_overlap_idx])
