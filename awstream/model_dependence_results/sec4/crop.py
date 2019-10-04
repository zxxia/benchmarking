from utils.model_utils import load_full_model_detection
# load_full_model_detection_new
from utils.utils import load_metadata
# from matplotlib import cm
# import matplotlib.pyplot as plt
import numpy as np
# import os
import pdb
import cv2


PATH = '/mnt/data/zhujun/dataset/Youtube/'
TEMPORAL_SAMPLING_LIST = [20, 15, 10, 5, 4, 3, 2.5, 2, 1.8, 1.5, 1.2, 1]
DATASET_LIST = sorted(['traffic', 'jp_hw', 'russia', 'tw_road', 'highway',
                       'tw_under_bridge', 'highway_normal_traffic', 'nyc',
                       'lane_split', 'tw', 'tw1', 'jp', 'russia1', 'park',
                       'driving_downtown', 'drift', 'crossroad4', 'driving1',
                       'crossroad3', 'crossroad2', 'crossroad', 'driving2',
                       'motorway'])
DATASET_LIST = ['motorway', 'park', 'highway', 'nyc', 'drift']
IMAGE_RESOLUTION_DICT = {'360p': [640, 360],
                         '480p': [854, 480],
                         '540p': [960, 540],
                         '576p': [1024, 576],
                         '720p': [1280, 720],
                         '1080p': [1920, 1080],
                         '3840p': [3840, 2160], }
SHORT_VIDEO_LENGTH = 30  # seconds
IOU_THRESH = 0.5
TARGET_F1 = 0.9
# TARGET_F1 = 0.8
PROFILE_LENGTH = 30  # seconds
OFFSET = 0  # 1*60+30
RESOLUTION_LIST = ['720p', '540p', '480p', '360p']  # '2160p', '1080p',

ORIGINAL_REOSL = '720p'
SELECTED_REOSL = '360p'
divisor = 6
def main():
    for dataset in DATASET_LIST:
        metadata = load_metadata(PATH + dataset + '/metadata.json')
        resolution = metadata['resolution']
        height = metadata['resolution'][1]
        # load detection results of fasterRCNN + full resolution +
        # highest frame rate as ground truth
        frame_rate = metadata['frame rate']
        frame_cnt = metadata['frame count']
        num_of_short_videos = frame_cnt//(SHORT_VIDEO_LENGTH*frame_rate)

        # gt_dict = defaultdict(None)
        # dt_dict = defaultdict(None)
        gt_file = PATH + dataset + '/' + '720p' + \
            '/profile/updated_gt_FasterRCNN_COCO.csv'
        gt, num_frames = load_full_model_detection(gt_file)
        print(frame_cnt)
        for j in range(1, frame_cnt+1):
            img_name = PATH + dataset + '/720p/{:06d}.jpg'.format(j)
            print(img_name)
            img = cv2.imread(img_name)
            mask = np.zeros_like(img, dtype=np.uint8)
            for box in gt[j]:
                xmin, ymin, xmax, ymax = box[:4]
                # print(ymin, ymax, xmax, xmax)
                mask[ymin:ymax, xmin:xmax] = 1
            img *= mask
            cv2.imwrite(PATH + dataset + "/720p/cropped/{:06d}.jpg".format(j), img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            # break
        # break
            # cv2.imshow('', img)
            # cv2.waitKey(0)
            # cv2.destroyWindow('')
            # cv2.destroyAllWindows()
            # break

if __name__ == '__main__':
    main()
