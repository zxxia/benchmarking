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
               'crossroad4', 'drift', 'highway', 'highway_normal_traffic',
               'jp', 'jp_hw', 'motorway', 'nyc', 'russia',
               'russia1', 'traffic', 'tw', 'tw1', 'tw_road',
               'tw_under_bridge', 't_crossroad'],
    'moving': ['driving1', 'driving2', 'driving_downtown', 'park',
               'lane_split', 'road_trip']
}


class COCOLabels(Enum):
    """COCO dataset object labels."""

    CAR = 3
    BUS = 6
    TRAIN = 7
    TRUCK = 8


MODEL_COST = {'mobilenet': 31,
              'Inception': 58,
              'FasterRCNN': 106}
