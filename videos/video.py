"""Definition of Video class."""
import copy
import glob
import os
import subprocess

import cv2

from benchmarking.constants import RESOL_DICT, COCOLabels
from benchmarking.utils.model_utils import (
    convert_detection_to_classification, filter_video_detections,
    load_full_model_detection, remove_overlappings, smooth_classification)


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
               target_frame_rate=None, save_video=True, crf=25):
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

        cmd = ['ffmpeg', '-y',  '-r',
               str(target_frame_rate), '-f', 'concat', '-safe', '0', '-i',
               tmp_list_file, '-s', frame_size,
               '-vcodec', 'libx264', '-crf', str(crf), '-pix_fmt',
               'yuv420p', '-hide_banner', output_video_name]
        subprocess.run(cmd, check=True)
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

        cmd = ['ffmpeg', '-y', '-loglevel', 'panic', '-r',
               str(target_frame_rate), '-f', 'concat', '-safe', '0', '-i',
               tmp_list_file, '-s', frame_size,
               '-vcodec', 'libx264', '-crf', str(self._quality_level),
               '-pix_fmt', 'yuv420p', '-force_key_frames',
               "expr:gte(t,n_forced*1)",
               '-hide_banner', output_video_name]
        subprocess.run(cmd, check=True)
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


class GeneralVideo(Video):
    def __init__(self, dataset_info, resolution_name, model='FasterRCNN', filter_flag=True,
                 merge_label_flag=False):
        """GeneralVideo Video Constructor."""

        frame_rate = dataset_info['frame_rate']
        dataset_path = dataset_info['path']
        if not dataset_path.endswith('/'):
            name = os.path.basename(dataset_path)
        else:
            name = dataset_path.split('/')[-2]

        detection_file = os.path.join(dataset_path, resolution_name, 'profile',
                                      'updated_gt_' + model + '_COCO_no_filter.csv')
        image_path = os.path.join(dataset_path, resolution_name)
        dets, num_of_frames = load_full_model_detection(detection_file)
        dets_nofilter = copy.deepcopy(dets)
        resolution = RESOL_DICT[resolution_name]
        if dataset_info['type'] == 'png':
            self.extension = 'png'
        else:
            self.extension = 'jpg'
        # TODO: handle overlapping boxes
        camera_type = dataset_info['camera_type']
        if camera_type == 'static':
            if filter_flag:
                dets, dropped_dets = filter_video_detections(
                    dets,
                    target_types={COCOLabels.CAR.value,
                                  COCOLabels.BUS.value,
                                  COCOLabels.TRUCK.value},
                    score_range=(0.3, 1.0),
                    width_range=(resolution[0] // 20, resolution[0]/2),
                    height_range=(resolution[1] // 20, resolution[1]/2))
                # self._dropped_detections = dropped_dets
                if merge_label_flag:
                    for frame_idx, boxes in dets.items():
                        for box_idx, _ in enumerate(boxes):
                            # Merge all cars and trucks into cars
                            dets[frame_idx][box_idx][4] = COCOLabels.CAR.value
                    #     dets[frame_idx] = remove_overlappings(boxes, 0.3)
            else:
                dropped_dets = None

        elif camera_type == 'moving':
            if filter_flag:
                dets, dropped_dets = filter_video_detections(
                    dets,
                    target_types={COCOLabels.CAR.value,
                                  COCOLabels.BUS.value,
                                  COCOLabels.TRUCK.value},
                    height_range=(resolution[1] // 20, resolution[1]))
                if merge_label_flag:
                    for frame_idx, boxes in dets.items():
                        for box_idx, _ in enumerate(boxes):
                            # Merge all cars and trucks into cars
                            dets[frame_idx][box_idx][4] = COCOLabels.CAR.value
                #     dets[frame_idx] = remove_overlappings(boxes, 0.3)
            else:
                dropped_dets = None

        if camera_type == 'road_trip':
            for frame_idx in dets:
                tmp_boxes = []
                for box in dets[frame_idx]:
                    xmin, ymin, xmax, ymax = box[:4]
                    if ymin >= 500/720*resolution[1] \
                            and ymax >= 500/720*resolution[1]:
                        continue
                    if (xmax - xmin) >= 2/3 * resolution[0]:
                        continue
                    tmp_boxes.append(box)
                dets[frame_idx] = tmp_boxes

        super().__init__(name, frame_rate, resolution, dets, dets_nofilter,
                         image_path, camera_type, model, dropped_dets)

    def get_frame_image_name(self, frame_index):
        """Return the image file name at frame index."""
        first_image = next(glob.iglob(
            self._image_path + "/*." + self.extension))
        filename_len = len(os.path.basename(first_image).split('.')[0])
        filename = str(frame_index).zfill(filename_len) + '.' + self.extension
        img_file = os.path.join(
            self._image_path, filename)
        return img_file
