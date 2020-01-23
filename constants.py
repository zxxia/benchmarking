"""This module contains some constants used accross different modules."""
from enum import Enum

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
    'static': ['crossroad', 'crossroad2', 'crossroad3',
               'crossroad4', 'crossroad5',
               'crossroad5_night', 'crossroad6', 'crossroad7',
                'drift', 'highway', 'highway1', 'highway_normal_traffic',
                'highway_no_traffic',
                'jp', 'jp_hw', 'motorway', 'russia',
               'russia1', 'traffic', 'tw', 'tw1', 'tw_road',
               'tw_under_bridge', 't_crossroad', 'canada_crossroad',
               'cropped_crossroad4', 'cropped_crossroad4_2', 'cropped_crossroad4_3',  'cropped_crossroad3',
               'cropped_crossroad5', 'cropped_crossroad5_night',
                'crossroad2_night'],
    'moving': ['driving1', 'driving2', 'driving_downtown', 'park', 'motor', 'nyc',
                'reckless_driving', 'street_racing',
               'lane_split', 'road_trip', 'cropped_driving1', 'cropped_driving2']
}


class COCOLabels(Enum):
    """COCO dataset object labels."""

    CAR = 3
    BUS = 6
    TRAIN = 7
    TRUCK = 8


MODEL_COST = {'mobilenet': 31,
              'inception': 58,
              'Inception': 58,
              'resnet50': 89,
              'Resnet50': 89,
              'FasterRCNN50': 89,
              'FasterRCNN': 106}

babygroot_DT_ROOT = '/mnt/data/zhujun/dataset/Youtube/'
babygroot_model_path = '/home/zhujunxiao/video_analytics_pipelines/models/research/object_detection_old/'
MODEL_PATH = {'FasterRCNN': 'faster_rcnn_resnet101_coco_2018_01_28',
              'FasterRCNN50': 'faster_rcnn_resnet50_coco_2018_01_28',
              'Inception': 'faster_rcnn_inception_v2_coco_2018_01_28',
              'mobilenet': 'ssd_mobilenet_v2_coco_2018_03_29'
}


