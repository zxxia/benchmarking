"""Definition of MOT15Video class."""
import copy

from constants import RESOL_DICT, COCOLabels
from utils.utils import (filter_video_detections, load_full_model_detection,
                         remove_overlappings)
from videos.video import Video


class MOT15Video(Video):
    """Class of MOT15Video."""

    # TODO: finishe the implementation

    FRAME_RATE_MAP = {
        'ADL-Rundle-6':  30,
        'ADL-Rundle-8': 30,
        'ETH-Bahnhof': 14,
        'ETH-Pedcross2': 14,
        'ETH-Sunnyday': 14,
        'KITTI-13': 10,
        'KITTI-17': 10,
        'PETS09-S2L1': 7,
        'TUD-Campus': 25,
        'TUD-Stadtmitte': 25,
        'Venice-2': 30,
        'ADL-Rundle-1': 30,
        'ADL-Rundle-3': 30,
        'AVG-TownCentre': 2.5,
        'ETH-Crossing': 14,
        'ETH-Jelmoli': 14,
        'ETH-Linthescher': 14,
        'KITTI-16': 10,
        'KITTI-19': 10,
        'PETS09-S2L2': 7,
        'TUD-Crossing': 25,
        'Venice-1': 30}

    def __init__(self, root, video_name, resolution_name,
                 detection_file, image_path,
                 model='FasterRCNN', filter_flag=True, merge_label_flag=False):
        """MOT15Video Constructor."""
        dets, num_of_frames = load_full_model_detection(detection_file)
        dets_nofilter = copy.deepcopy(dets)
        # resolution = (1242, 375)
        resolution = RESOL_DICT[resolution_name]
        frame_rate = self.FRAME_RATE_MAP[video_name]
        if filter_flag:
            dets, dropped_dets = filter_video_detections(
                dets,
                target_types={COCOLabels.CAR.value,
                              COCOLabels.BUS.value,
                              COCOLabels.TRUCK.value},
                score_range=(0.3, 1.0),
                width_range=(resolution[0] // 20, resolution[0]/2),
                height_range=(resolution[1] // 20, resolution[1]))
            self._dropped_detections = dropped_dets
            if merge_label_flag:
                for frame_idx, boxes in dets.items():
                    for box_idx, _ in enumerate(boxes):
                        # Merge all cars and trucks into cars
                        dets[frame_idx][box_idx][4] = COCOLabels.CAR.value
                #     dets[frame_idx] = remove_overlappings(boxes, 0.3)
        else:
            dropped_dets = None
        super().__init__(video_name, frame_rate, resolution, dets,
                         dets_nofilter, image_path, 'moving', model,
                         dropped_dets)
