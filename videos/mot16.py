"""Definition of MOT16Video class."""
import configparser
import copy
import os

from constants import RESOL_DICT, COCOLabels
from utils.utils import (filter_video_detections, remove_overlappings)
from videos.video import Video
from object_detection.infer import load_object_detection_results


class MOT16Video(Video):
    """Class of MOT16Video."""

    def __init__(self, root, video_name, resolution_name,
                 model='faster_rcnn_resnet101', qp=23, filter_flag=True,
                 merge_label_flag=False,
                 classes_interested={COCOLabels.CAR.value,
                                     COCOLabels.BUS.value,
                                     COCOLabels.TRUCK.value}, cropped=False):
        """MOT16Video Constructor."""
        resolution = RESOL_DICT[resolution_name]
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
        config = configparser.ConfigParser()
        seqinfo_path = os.path.join(root, 'seqinfo.ini')
        config.read(seqinfo_path)
        frame_rate = int(config['Sequence']['frameRate'])
        if filter_flag:
            dets, dropped_dets = filter_video_detections(
                dets,
                target_types=classes_interested,
                score_range=(0.3, 1.0),
                # TODO: this part need to be revised
                # width_range=(resolution[0] // 20, resolution[0]/2),
                # width_range=(resolution[0] // 20, resolution[0]/2),
                height_range=(resolution[1] // 20, resolution[1])
            )
            self._dropped_detections = dropped_dets
            #     dets[frame_idx] = remove_overlappings(boxes, 0.3)
        else:
            dropped_dets = None
        if merge_label_flag:
            for frame_idx, boxes in dets.items():
                for box_idx, _ in enumerate(boxes):
                    # Merge all cars and trucks into cars
                    # dets[frame_idx][box_idx][4] = COCOLabels.CAR.value
                    dets[frame_idx][box_idx][4] = min(classes_interested)
        super().__init__(video_name, frame_rate, resolution, dets,
                         dets_nofilter, image_path, 'moving', model,
                         dropped_dets)

    def get_frame_image_name(self, frame_index):
        """Return the image file name at frame index."""
        img_file = os.path.join(
            self._image_path, '{:06d}.jpg'.format(frame_index))
        return img_file
