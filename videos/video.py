"""Definition of Video class."""
import os
import subprocess

import cv2

# from benchmarking.constants import RESOL_DICT, COCOLabels
from utils.utils import smooth_classification, \
    convert_detection_to_classification

# , filter_video_detections,
#     load_full_model_detection, remove_overlappings, )


class Video:
    """Base class of Video."""

    def __init__(self, name, frame_rate, resolution, detections,
                 detections_nofilter, image_path,
                 video_type, model, dropped_detections=None):
        """Constructor."""
        self._name = name
        self._frame_rate = frame_rate
        self._resolution = resolution
        self._detections_nofilter = detections_nofilter
        self._detections = detections
        self._dropped_detections = dropped_detections
        self._frame_count = len(detections)
        self._image_path = image_path
        self._video_type = video_type
        self._model = model
        self._quality_level = 23

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
        return self.quality_level

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

    @property
    def image_path(self):
        """Return detection model type."""
        return self._image_path

    def get_frame_image(self, frame_index):
        """Return the image at frame index."""
        img = cv2.imread(self.get_frame_image_name(frame_index))

        if img.shape[0] != self._resolution[1] or \
                img.shape[1] != self._resolution[0]:
            img = cv2.resize(img, self._resolution,
                             interpolation=cv2.INTER_AREA)
        return img

    def get_frame_image_name(self, frame_index):
        """Return the image file name at frame index."""
        return None

    def get_frame_filesize(self, frame_index):
        """Return the image file size at frame index."""
        filename = self.get_frame_image_name(frame_index)
        return os.path.getsize(filename)

    def get_frame_detection(self, frame_index):
        """Return the object detections at frame index."""
        return self._detections[frame_index]

    def get_dropped_frame_detection(self, frame_index):
        """Return the object detections which are dropped at frame index."""
        if self._dropped_detections is None:
            return None
        return self._dropped_detections[frame_index]

    def get_dropped_video_detection(self):
        """Return the dropped object detections of the video by filter."""
        return self._dropped_detections

    def get_video_detection(self):
        """Return the object detections of the video."""
        return self._detections

    def get_video_classification_label(self):
        """Return the classification label for each video frame."""
        classification_labels = convert_detection_to_classification(
            self._detections_nofilter, self._resolution)
        smoothed_classification_labels = smooth_classification(
            classification_labels)
        return smoothed_classification_labels

    def encode(self, output_video_name, target_frame_indices=None,
               target_frame_rate=None, save_video=True, crf=23):
        """Encode the target frames into a video and return video size."""
        if os.path.exists(output_video_name):
            return os.path.getsize(output_video_name)
        print("start generating "+output_video_name)
        tmp_list_file = output_video_name + '_list.txt'
        with open(tmp_list_file, 'w') as f_list:
            for i in target_frame_indices:
                # based on sample rate, decide whether this frame is sampled
                line = 'file \'{}\'\n'.format(self.get_frame_image_name(i))
                f_list.write(line)

        # compress the sampled image into a video
        frame_size = '{}x{}'.format(self._resolution[0], self._resolution[1])

        cmd = 'ffmpeg -y -r {} -f concat -safe 0 -i {} -s {} -vcodec libx264 '\
            '-crf {} -pix_fmt yuv420p -hide_banner {}'.format(
                target_frame_rate, tmp_list_file, frame_size, crf,
                output_video_name)
        subprocess.run(cmd.split(' '), check=True)
        # get the video size
        video_size = os.path.getsize(output_video_name)
        if not save_video:
            os.remove(output_video_name)
        os.remove(tmp_list_file)
        print('target fps={}, target resolution={}, video size={}'
              .format(target_frame_rate, self._resolution, video_size))
        print('finish generating {} and size={}'.format(
            output_video_name, video_size))
        return video_size

    def encode_iframe_control(self, output_video_name,
                              target_frame_indices=None,
                              target_frame_rate=None, save_video=True):
        """Encode the target frames into a video and return video size.

        Encode I-frame every second.
        """
        if os.path.exists(output_video_name):
            return os.path.getsize(output_video_name)
        print("start generating "+output_video_name)
        tmp_list_file = output_video_name + '_list.txt'
        with open(tmp_list_file, 'w') as f_list:
            for i in target_frame_indices:
                # based on sample rate, decide whether this frame is sampled
                line = 'file \'{}\'\n'.format(self.get_frame_image_name(i))
                f_list.write(line)

        # compress the sampled image into a video
        frame_size = str(self._resolution[0]) + 'x' + str(self._resolution[1])

        cmd = 'ffmpeg -y -loglevel panic -r {} -f concat -safe 0 -i {} -s {} '\
            '-vcodec libx264 -crf {} -pix_fmt yuv420p -force_key_frames ' \
            'expr:gte(t,n_forced*1) -hide_banner {}'.format(
                target_frame_rate, tmp_list_file, frame_size,
                self._quality_level, output_video_name)
        subprocess.run(cmd.split(' '), check=True)
        # get the video size
        video_size = os.path.getsize(output_video_name)
        if not save_video:
            os.remove(output_video_name)
        os.remove(tmp_list_file)
        print('target fps={}, target resolution={}, video size={}'
              .format(target_frame_rate, self._resolution, video_size))
        print('finish generating {} and size={}'.format(
            output_video_name, video_size))
        return video_size
