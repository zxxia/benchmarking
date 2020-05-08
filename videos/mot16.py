"""Definition of MOT16Video class."""
import configparser
import copy
import os

from constants import RESOL_DICT, COCOLabels
from utils.utils import (filter_video_detections, load_full_model_detection,
                         remove_overlappings)
from videos.video import Video


class MOT16Video(Video):
    """Class of MOT16Video."""

    # TODO: finish MOT16 dataset abstraction

    def __init__(self, root, video_name, resolution_name,
                 detection_file, image_path,
                 model='FasterRCNN', filter_flag=True, merge_label_flag=False):
        """MOT16Video Constructor."""
        dets, num_of_frames = load_full_model_detection(detection_file)
        dets_nofilter = copy.deepcopy(dets)
        # resolution = (1242, 375)
        resolution = RESOL_DICT[resolution_name]
        config = configparser.ConfigParser()
        seqinfo_path = os.path.join(root, 'seqinfo.ini')
        config.read(seqinfo_path)
        frame_rate = int(config['Sequence']['frameRate'])
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
