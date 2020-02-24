"""Definition of KittiVideo class."""
import copy
import os

from benchmarking.constants import RESOL_DICT, COCOLabels
from benchmarking.utils.model_utils import (filter_video_detections,
                                            load_full_model_detection,
                                            remove_overlappings)
from video import Video


class KittiVideo(Video):
    """Class of KittiVideo."""

    def __init__(self, name, resolution_name, detection_file, image_path,
                 model='FasterRCNN', filter_flag=True, merge_label_flag=False):
        """Kitti Video Constructor."""
        dets, num_of_frames = load_full_model_detection(detection_file)
        dets_nofilter = copy.deepcopy(dets)
        if resolution_name is None:
            resolution = (1242, 375)
        else:
            resolution = RESOL_DICT[resolution_name]
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
        super().__init__(name, 10, resolution, dets, dets_nofilter,
                         image_path, 'moving', model, dropped_dets)

    def get_frame_image_name(self, frame_index):
        """Return the image file name at frame index."""
        img_file = os.path.join(
            self._image_path, '{:010d}.png'.format(frame_index))
        return img_file
