"""This module contains some constants used accross different modules."""
from enum import Enum
import numpy as np
RESOL_DICT = {'180p': (320, 180),
              '240p': (426, 240),
              '300p': (534, 300),
              '360p': (640, 360),
              '375p': (1242, 375),
              '480p': (854, 480),
              '540p': (960, 540),
              '576p': (1024, 576),
              '720p': (1280, 720),
              '1080p': (1920, 1080),
              '2160p': (3840, 2160)}

CAMERA_TYPES = {
    'static': ['crossroad', 'crossroad2', 'crossroad3', 'crossroad4',
               'crossroad5', 'crossroad5_night', 'crossroad6', 'crossroad7',
               'drift', 'highway', 'highway1', 'highway_normal_traffic',
               'highway_no_traffic', 'jp', 'jp_hw', 'motorway', 'russia',
               'russia1', 'traffic', 'tw', 'tw1', 'tw_road', 'tw_under_bridge',
               't_crossroad', 'canada_crossroad', 'cropped_crossroad4',
               'cropped_crossroad4_2', 'cropped_crossroad4_3',
               'cropped_crossroad3', 'cropped_crossroad5',
               'cropped_crossroad5_night', 'crossroad2_night'],
    'moving': ['driving1', 'driving2', 'driving_downtown', 'park', 'motor',
               'nyc', 'reckless_driving', 'street_racing', 'lane_split',
               'road_trip', 'cropped_driving1', 'cropped_driving2']
}


class COCOLabels(Enum):
    """COCO dataset object labels."""

    PERSON = 1
    CAR = 3
    BUS = 6
    TRAIN = 7
    TRUCK = 8


OFFSET = 0

MODEL_COST = {'mobilenet': 31,
              'inception': 58,
              'Inception': 58,
              'resnet50': 89,
              'Resnet50': 89,
              'FasterRCNN50': 89,
              'FasterRCNN': 106,
              'faster_rcnn_resnet101': 106,
              }


RESOL_LIST = ['360p', '480p', '540p', '720p']
# RESOL_LIST = ['720p']
MODEL_LIST = ['FasterRCNN', 'mobilenet', 'Inception', 'FasterRCNN50']

Original_resol = '720p'
Full_model = 'FasterRCNN'


Glimpse_para1_dict = {
    'crossroad': np.arange(30, 42, 2),
    'crossroad2': np.arange(20, 42, 2),
    'crossroad3': np.arange(70, 100, 3),
    'crossroad4': np.arange(30, 62, 2),
    'drift': np.arange(290, 400, 10),
    'driving1': np.arange(10, 25, 2),
    'driving2': np.arange(5, 30, 2),
    'driving_downtown': np.arange(4, 20, 2),
    'highway': np.arange(30, 40, 2),
    'highway_normal_traffic': np.arange(34, 40, 2),
    'jp': np.arange(30, 40, 2),
    'jp_hw': np.arange(30, 40, 2),
    'lane_split': np.arange(6, 14, 2),
    'motorway': np.arange(2, 6, 2),
    'nyc': np.arange(2, 22, 2),
    'park': np.arange(2, 10, 2),
    'russia': np.arange(100, 400, 20),
    'russia1': np.arange(100, 400, 20),
    'traffic': np.arange(6, 15, 1),
    'tw': np.arange(25, 55, 5),
    'tw1': np.arange(25, 55, 5),
    'tw_road': np.arange(15, 45, 5),
    'tw_under_bridge': np.arange(350, 450, 10),
    'waymo': np.arange(20, 220, 20),
    'video': np.arange(290, 400, 10)
}

Glimpse_para2_list = [1]
