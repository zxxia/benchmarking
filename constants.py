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
               'tw_under_bridge', 't_crossroad', 'canada_crossroad',
               'cropped_crossroad4', 'cropped_crossroad4_2', 'cropped_crossroad5', 'crossroad2_night'],
    'moving': ['driving1', 'driving2', 'driving_downtown', 'park',
               'lane_split', 'road_trip', 'cropped_driving2']
}


class COCOLabels(Enum):
    """COCO dataset object labels."""

    CAR = 3
    BUS = 6
    TRAIN = 7
    TRUCK = 8


MODEL_COST = {'mobilenet': 31,
              'inception': 58,
              'resnet50': 89,
              'Resnet50': 89,
              'FasterRCNN50': 89,
              'FasterRCNN': 106}


def load_COCOlabelmap(label_map_path):
    COCO_Labelmap = {}
    with open(label_map_path, 'r') as f:
        line = f.readline()
        while line:
            if 'id' in line:
                ID = int(line.strip().split(':')[1].strip())
                line = f.readline()
                label = line.strip().split(':')[1]
                COCO_Labelmap[ID] = label.strip().replace('"', '')
                line = f.readline()
            else:
                line = f.readline()

    return COCO_Labelmap
