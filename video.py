"""Definition of videos."""
import os
import subprocess
import cv2
from utils.model_utils import load_full_model_detection, \
    filter_video_detections
from utils.utils import load_metadata
from constants import CAMERA_TYPES, COCOLabels, RESOL_DICT


class Video:
    """Base class of Video."""

    def __init__(self, name, frame_rate, resolution, detections, image_path,
                 video_type, model):
        """Constructor."""
        self._name = name
        self._frame_rate = frame_rate
        self._resolution = resolution
        self._detections = detections
        self._frame_count = len(detections)
        self._image_path = image_path
        self._video_type = video_type
        self._model = model

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
    def duration(self):
        """Return video duration in seconds."""
        return self._frame_count / self._frame_rate

    @property
    def model(self):
        """Return detection model type."""
        return self._model

    def get_frame_image(self, frame_index):
        """Return the image at frame index."""
        return None

    def get_frame_detection(self, frame_index):
        """Return the object detections at frame index."""
        return self._detections[frame_index]

    def get_video_detection(self):
        """Return the object detections of the video."""
        return self._detections

    def encode(self, output_video_name, target_frame_indices=None,
               target_frame_rate=None):
        """Encode the frames into a video and return output video size."""
        return 0

    # def compute_object_size():
    #     # TODO: consider implementing the feature computation in video
    #     return 0


class YoutubeVideo(Video):
    """Class of YoutubeVideo."""

    def __init__(self, name, resolution_name, metadata_file, detection_file,
                 image_path, model='FasterRCNN'):
        """Youtube Video Constructor."""
        metadata = load_metadata(metadata_file)
        frame_rate = metadata['frame rate']
        dets, num_of_frames = load_full_model_detection(detection_file)
        resolution = RESOL_DICT[resolution_name]

        # TODO: handle overlapping boxes
        if name in CAMERA_TYPES['static']:
            camera_type = 'static'
            dets, dropped_dets = filter_video_detections(
                dets,
                target_types={COCOLabels.CAR.value,
                              COCOLabels.BUS.value,
                              COCOLabels.TRUCK.value},
                width_range=(0, resolution[0]/2),
                height_range=(0, resolution[1] / 2))
            self._dropped_detections = dropped_dets
        elif name in CAMERA_TYPES['moving']:
            camera_type = 'moving'
            dets, dropped_dets = filter_video_detections(
                dets,
                target_types={COCOLabels.CAR.value,
                              COCOLabels.BUS.value,
                              COCOLabels.TRUCK.value},
                height_range=(resolution[1] // 20, resolution[1]))
            self._dropped_detections = dropped_dets
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
                         image_path, camera_type, model)

    def get_frame_image(self, frame_index):
        """Return the image at frame index."""
        img_name = format(frame_index, '06d') + '.jpg'
        img_file = os.path.join(self._image_path, img_name)
        img = cv2.imread(img_file)
        return img

    def get_dropped_frame_detection(self, frame_index):
        """Return the object detections which are dropped at frame index."""
        return self._dropped_detections[frame_index]

    def get_dropped_video_detection(self):
        """Return the dropped object detections of the video by filter."""
        return self._dropped_detections

    def encode(self, output_video_name, target_frame_indices=None,
               target_frame_rate=None):
        """Encode the target frames into a video and ."""
        print("start generating "+output_video_name)
        tmp_list_file = output_video_name + '_list.txt'
        with open(tmp_list_file, 'w') as f_list:
            for i in target_frame_indices:
                # based on sample rate, decide whether this frame is sampled
                line = 'file \'{}/{:06}.jpg\'\n'.format(self._image_path, i)
                f_list.write(line)

        # compress the sampled image into a video
        frame_size = str(self._resolution[0]) + 'x' + str(self._resolution[1])

        cmd = ['ffmpeg', '-y', '-loglevel', 'panic', '-r',
               str(target_frame_rate), '-f', 'concat', '-safe', '0', '-i',
               tmp_list_file, '-s', frame_size,
               '-vcodec', 'libx264', '-crf', '25', '-pix_fmt',
               'yuv420p', '-hide_banner', output_video_name]
        subprocess.run(cmd, check=True)
        # get the video size
        video_size = os.path.getsize(output_video_name)
        # os.remove(output_video)
        os.remove(tmp_list_file)
        print('target fps={}, target resolution={}, video size={}'
              .format(target_frame_rate, self._resolution, video_size))
        print('finish generating {} and size={}'.format(
            output_video_name, video_size))
        return video_size


class KittiVideo(Video):
    """Class of KittiVideo."""

    def __init__(self, name, detection_file, image_path, model='frcnn',
                 filter_flag=False):
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
        super().__init__(name, 10, resolution, dets, image_path, 'moving',
                         model)

    def get_frame_image(self, frame_index):
        """Return the image at frame index."""
        img_name = format(frame_index, '010d') + '.png'
        img_file = os.path.join(self._image_path, img_name)
        img = cv2.imread(img_file)
        return img


class WaymoVideo(Video):
    """Class of WaymoVideo."""

    def __init__(self, name, detection_file, image_path, model='frcnn'):
        """Kitti Video Constructor."""
        dets, num_of_frames = load_full_model_detection(detection_file)
        super().__init__(name, 20, (1272, 375), dets, image_path, 'moving',
                         model)
