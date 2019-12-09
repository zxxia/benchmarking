"""Definition of videos."""
import os
from utils.model_utils import load_full_model_detection, \
    filter_video_detections
from utils.utils import load_metadata
from constants import CAMERA_TYPES, COCOLabels, RESOL_DICT
import cv2


class Video:
    """Base class of Video."""

    def __init__(self, name, frame_rate, resolution, detections, image_path,
                 video_type):
        """Constructor."""
        self._name = name
        self._frame_rate = frame_rate
        self._resolution = resolution
        self._detections = detections
        self._frame_count = len(detections)
        self._image_path = image_path
        self._video_type = video_type

    @property
    def video_name(self):
        """Return video name."""
        return self._name

    @property
    def frame_rate(self):
        """Return video frame rate(fps)."""
        return self._frame_rate

    @property
    def resolution(self):
        """Return video resolution."""
        return self._resolution

    @property
    def frame_count(self):
        """Return number of frames in the video."""
        return self._frame_count

    @property
    def quality_level(self):
        """Return quality level of the video."""
        # TODO: finish quality level
        return 0

    @property
    def start_frame_index(self):
        """Return the minimum frame index of the video."""
        return min(self._detections)

    @property
    def end_frame_index(self):
        """Return the maximum frame index of the video."""
        return max(self._detections)

    @property
    def video_type(self):
        """Return video type."""
        return self._video_type

    @property
    def get_video_duration(self):
        """Return video duration in seconds."""
        return self._frame_count / self._frame_rate

    def get_frame_image(self, frame_index):
        """Return the image at frame index."""
        return None

    def get_frame_detection(self, frame_index):
        """Return the object detections at frame index."""
        return self._detections[frame_index]

    def get_video_detection(self):
        """Return the object detections at frame index."""
        return self._detections

    def encode(self):
        # TODO: finish the encoding logic
        """Encode the frames into a video."""
        return None


class YoutubeVideo(Video):
    """Class of YoutubeVideo."""

    def __init__(self, name, resolution_name, metadata_file, detection_file,
                 image_path, filter_flag=False):
        """Youtube Video Constructor."""
        metadata = load_metadata(metadata_file)
        frame_rate = metadata['frame rate']
        dets, num_of_frames = load_full_model_detection(detection_file)
        resolution = RESOL_DICT[resolution_name]

        if name in CAMERA_TYPES['static']:
            camera_type = 'static'
            if filter_flag:  # doing bboxes filtering
                dets = filter_video_detections(
                    dets,
                    target_types={COCOLabels.CAR.value,
                                  COCOLabels.BUS.value,
                                  COCOLabels.TRUCK.value},
                    width_range=(0, resolution[0]/2),
                    height_range=(0, resolution[1] / 2))
        elif name in CAMERA_TYPES['moving']:
            camera_type = 'moving'
            if filter_flag:  # doing bboxes filtering
                dets = filter_video_detections(
                    dets,
                    target_types={COCOLabels.CAR.value,
                                  COCOLabels.BUS.value,
                                  COCOLabels.TRUCK.value},
                    height_range=(resolution[1] // 20, resolution[1]))
        # TODO: need to handle roadtrip
        # if name == 'road_trip':
        #     for frame_idx in dets:
        #         tmp_boxes = []
        #         for box in dets[frame_idx]:
        #             xmin, ymin, xmax, ymax = box[:4]
        #             if ymin >= 500 and ymax >= 500:
        #                 continue
        #             if (xmax - xmin) >= 2/3 * 1280:
        #                 continue
        #             tmp_boxes.append(box)
        #         dets[frame_idx] = tmp_boxes

        super().__init__(name, frame_rate, resolution, dets,
                         image_path, camera_type)

    def get_frame_image(self, frame_index, is_gray_scale=False):
        """Return the image at frame index."""
        img_name = format(frame_index, '06d') + '.jpg'
        img_file = os.path.join(self.image_path, img_name)
        if is_gray_scale:
            img = cv2.imread(img_file, 0)
        else:
            img = cv2.imread(img_file)
        return img


class KittiVideo(Video):
    """Class of KittiVideo."""

    def __init__(self, name, detection_file, image_path, filter_flag=False):
        """Kitti Video Constructor."""
        dets, num_of_frames = load_full_model_detection(detection_file)
        resolution = (1242, 375)
        if filter_flag:
            dets = filter_video_detections(
                dets,
                target_types={COCOLabels.CAR.value,
                              COCOLabels.BUS.value,
                              COCOLabels.TRUCK.value},
                height_range=(resolution[1] // 20, resolution[1]))
        super().__init__(name, 10, resolution, dets, image_path, 'moving')

    def get_frame_image(self, frame_index):
        """Return the image at frame index."""
        img_name = format(frame_index, '010d') + '.png'
        img_file = os.path.join(self.image_path, img_name)
        img = cv2.imread(img_file)
        return img


class WaymoVideo(Video):
    """Class of WaymoVideo."""

    def __init__(self, name, detection_file, image_path):
        """Kitti Video Constructor."""
        dets, num_of_frames = load_full_model_detection(detection_file)
        super().__init__(name, 20, (1272, 375), dets, image_path, 'moving')
