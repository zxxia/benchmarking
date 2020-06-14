"""Definition of MOT15Video class."""
import copy
import os

from constants import RESOL_DICT, COCOLabels
from object_detection.infer import load_object_detection_results
from utils.utils import filter_video_detections, remove_overlappings
# load_full_model_detection,

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
                 model='faster_rcnn_resnet101', qp=23, filter_flag=True,
                 merge_label_flag=False,
                 classes_interested={COCOLabels.CAR.value,
                                     COCOLabels.BUS.value,
                                     COCOLabels.TRUCK.value}, cropped=False):
        """MOT15Video Constructor."""
        resolution = RESOL_DICT[resolution_name]
        # dets, num_of_frames = load_full_model_detection(detection_file)
        if cropped:
            image_path = os.path.join(root, resolution_name+'_cropped')
            detection_file = os.path.join(
                root, 'profile',
                f"{model}_{resolution[0]}x{resolution[1]}_{qp}_cropped_smoothed_detections.csv")
        else:
            image_path = os.path.join(root, resolution_name)
            detection_file = os.path.join(
                root, 'profile',
                f"{model}_{resolution[0]}x{resolution[1]}_{qp}_smoothed_detections.csv")
        print('loading {}...'.format(detection_file))
        dets = load_object_detection_results(detection_file)
        dets_nofilter = copy.deepcopy(dets)
        frame_rate = self.FRAME_RATE_MAP[video_name]
        if filter_flag:
            dets, dropped_dets = filter_video_detections(
                dets,
                target_types=classes_interested,
                score_range=(0.3, 1.0),
                width_range=(resolution[0] // 20, resolution[0]/2),
                height_range=(resolution[1] // 20, resolution[1]))
            self._dropped_detections = dropped_dets
            if merge_label_flag:
                for frame_idx, boxes in dets.items():
                    for box_idx, _ in enumerate(boxes):
                        # Merge all cars and trucks into cars
                        dets[frame_idx][box_idx][4] = min(classes_interested)
                #     dets[frame_idx] = remove_overlappings(boxes, 0.3)
        else:
            dropped_dets = None
        super().__init__(video_name, frame_rate, resolution, dets,
                         dets_nofilter, image_path, 'moving', model,
                         dropped_dets)
