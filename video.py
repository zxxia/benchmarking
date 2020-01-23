"""Definition of videos."""
import copy
import configparser
import os
import pdb
import subprocess
import cv2
import pandas as pd
from benchmarking.utils.model_utils import load_full_model_detection, \
    filter_video_detections, convert_detection_to_classification, smooth_classification
from benchmarking.constants import CAMERA_TYPES, COCOLabels, RESOL_DICT
from benchmarking.utils.utils import load_metadata
from benchmarking.utils.model_utils import remove_overlappings


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

    # def get_frame_image(self, frame_index):
    #     """Return the image at frame index."""
    #     return None
    def get_frame_image(self, frame_index):
        """Return the image at frame index."""
        img = cv2.imread(self.get_frame_image_name(frame_index))
        return img

    def get_frame_image_name(self, frame_index):
        """Return the image file name at frame index."""
        return None

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

    # def encode(self, output_video_name, target_frame_indices=None,
    #            target_frame_rate=None):
    #     """Encode the frames into a video and return output video size."""
    #     return 0

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
            video_size = os.path.getsize(output_video_name)
        else:
            print("start generating "+output_video_name)
            tmp_list_file = output_video_name + '_list.txt'
            with open(tmp_list_file, 'w') as f_list:
                for i in target_frame_indices:
                    # based on sample rate, decide whether this frame is sampled
                    line = 'file \'{}\'\n'.format(self.get_frame_image_name(i))
                    f_list.write(line)

            # compress the sampled image into a video
            frame_size = str(self._resolution[0]) + 'x' + str(self._resolution[1])

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


class YoutubeVideo(Video):
    """Class of YoutubeVideo."""

    def __init__(self, name, resolution_name, metadata_file, detection_file,
                 image_path, model='FasterRCNN', filter_flag=True,
                 merge_label_flag=False):
        """Youtube Video Constructor."""
        if metadata_file is not None:
            metadata = load_metadata(metadata_file)
            frame_rate = metadata['frame rate']
        else:
            frame_rate = 30
        dets, num_of_frames = load_full_model_detection(detection_file)
        dets_nofilter = copy.deepcopy(dets)
        resolution = RESOL_DICT[resolution_name]

        # TODO: handle overlapping boxes
        if name in CAMERA_TYPES['static']:
            camera_type = 'static'
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

        elif name in CAMERA_TYPES['moving']:
            camera_type = 'moving'
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

        if name == 'road_trip':
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

    # def get_frame_image(self, frame_index):
    #     """Return the image at frame index."""
    #     img = cv2.imread(self.get_frame_image_name(frame_index))
    #     return img

    def get_frame_image_name(self, frame_index):
        """Return the image file name at frame index."""
        img_file = os.path.join(
            self._image_path, '{:06d}.jpg'.format(frame_index))
        return img_file


    def get_frame_filesize(self, frame_index):
        filename = self.get_frame_image_name(frame_index)
        return os.path.getsize(filename)


    # def encode(self, output_video_name, target_frame_indices=None,
    #            target_frame_rate=None, save_video=True):
    #     """Encode the target frames into a video and return video size."""
    #     print("start generating "+output_video_name)
    #     tmp_list_file = output_video_name + '_list.txt'
    #     with open(tmp_list_file, 'w') as f_list:
    #         for i in target_frame_indices:
    #             # based on sample rate, decide whether this frame is sampled
    #             line = 'file \'{}/{:06}.jpg\'\n'.format(self._image_path, i)
    #             f_list.write(line)
    #
    #     # compress the sampled image into a video
    #     frame_size = str(self._resolution[0]) + 'x' + str(self._resolution[1])
    #
    #     cmd = ['ffmpeg', '-y', '-loglevel', 'panic', '-r',
    #            str(target_frame_rate), '-f', 'concat', '-safe', '0', '-i',
    #            tmp_list_file, '-s', frame_size,
    #            '-vcodec', 'libx264', '-crf', str(self._quality_level),
    #            '-pix_fmt', 'yuv420p', '-hide_banner', output_video_name]
    #     subprocess.run(cmd, check=True)
    #     # get the video size
    #     video_size = os.path.getsize(output_video_name)
    #     if not save_video:
    #         os.remove(output_video_name)
    #     os.remove(tmp_list_file)
    #     print('target fps={}, target resolution={}, video size={}'
    #           .format(target_frame_rate, self._resolution, video_size))
    #     print('finish generating {} and size={}'.format(
    #         output_video_name, video_size))
    #     return video_size

    def encode_iframe_control(self, output_video_name,
                              target_frame_indices=None,
                              target_frame_rate=None, save_video=True):
        """Encode the target frames into a video and return video size."""
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
               '-vcodec', 'libx264', '-crf', str(self._quality_level),
               '-pix_fmt', 'yuv420p', '-force_key_frames',
               "expr:gte(t,n_forced*2)",
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

    # a better way to create a video
    def encode_ffmpeg(self, video, frame_range, every_n_frame, output_path):
        """Create a video using ffmepg."""
        # TODO: add path to original video
        output_video_name = os.path.join(output_path, "{}.mp4".format(video))
        frame_size = str(self._resolution[0]) + 'x' + str(self._resolution[1])
        cmd = ["ffmpeg", "-y", "-i", video, '-an', '-vf',
               'select=between(n\,{}\,{})*not(mod(n\,{})),'
               'setpts=PTS-STARTPTS'.format(frame_range[0],
                                            frame_range[1], every_n_frame),
               '-vsync', 'vfr', '-s', frame_size, '-crf',
               str(self._quality_level), output_video_name, "-hide_banner"]
        print(cmd)
        subprocess.run(cmd, check=True)


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

    # def get_frame_image(self, frame_index):
    #     """Return the image at frame index."""
    #     img_name = format(frame_index, '010d') + '.png'
    #     img_file = os.path.join(self._image_path, img_name)
    #     img = cv2.imread(img_file)
    #     # img = cv2.resize(img, self._resolution, interpolation=cv2.INTER_AREA)
    #     return img

    def get_frame_image_name(self, frame_index):
        """Return the image file name at frame index."""
        img_file = os.path.join(
            self._image_path, '{:010d}.png'.format(frame_index))
        return img_file



    def get_frame_filesize(self, frame_index):
        filename = self.get_frame_image_name(frame_index)
        return os.path.getsize(filename)
    # def encode(self, output_video_name, target_frame_indices=None,
    #            target_frame_rate=None, save_video=True):
    #     """Encode the target frames into a video and return video size."""
    #     print("start generating "+output_video_name)
    #     tmp_list_file = output_video_name + '_list.txt'
    #     with open(tmp_list_file, 'w') as f_list:
    #         for i in target_frame_indices:
    #             # based on sample rate, decide whether this frame is sampled
    #             line = 'file \'{}\'\n'.format(self.get_frame_image_name(i))
    #             f_list.write(line)
    #
    #     # compress the sampled image into a video
    #     frame_size = str(self._resolution[0]) + 'x' + str(self._resolution[1])
    #
    #     cmd = ['ffmpeg', '-y',  '-r',
    #            str(target_frame_rate), '-f', 'concat', '-safe', '0', '-i',
    #            tmp_list_file, '-s', frame_size,
    #            '-vcodec', 'libx264', '-crf', '25', '-pix_fmt',
    #            'yuv420p', '-hide_banner', output_video_name]
    #     subprocess.run(cmd, check=True)
    #     # get the video size
    #     video_size = os.path.getsize(output_video_name)
    #     if not save_video:
    #         os.remove(output_video_name)
    #     os.remove(tmp_list_file)
    #     print('target fps={}, target resolution={}, video size={}'
    #           .format(target_frame_rate, self._resolution, video_size))
    #     print('finish generating {} and size={}'.format(
    #         output_video_name, video_size))
    #     return video_size


class WaymoVideo(Video):
    """Class of WaymoVideo."""

    def __init__(self, name, resolution_name, detection_file, image_path,
                 model='FasterRCNN', filter_flag=True, merge_label_flag=False):
        """Waymo Video Constructor."""
        dets, num_of_frames = load_full_model_detection(detection_file)
        dets_nofilter = copy.deepcopy(dets)
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
            # self._dropped_detections = dropped_dets
            if merge_label_flag:
                for frame_idx, boxes in dets.items():
                    for box_idx, _ in enumerate(boxes):
                        # Merge all cars and trucks into cars
                        dets[frame_idx][box_idx][4] = COCOLabels.CAR.value
            #     dets[frame_idx] = remove_overlappings(boxes, 0.3)
        else:
            dropped_dets = None

        super().__init__(name, 10, resolution, dets, dets_nofilter, image_path,
                         'moving', model, dropped_detections=dropped_dets)

    # def get_frame_image(self, frame_index):
    #     """Return the image at frame index."""
    #     img_name = format(frame_index, '04d') + '.jpg'
    #     img_file = os.path.join(self._image_path, img_name)
    #     img = cv2.imread(img_file)
    #     img = cv2.resize(img, self._resolution, interpolation=cv2.INTER_AREA)
    #     return img

    def get_frame_image_name(self, frame_index):
        """Return the image file name at frame index."""
        img_file = os.path.join(
            self._image_path, '{:04d}.jpg'.format(frame_index))
        return img_file

    # def encode(self, output_video_name, target_frame_indices=None,
    #            target_frame_rate=None, save_video=True):
    #     """Encode the target frames into a video and return video size."""
    #     print("start generating "+output_video_name)
    #     tmp_list_file = output_video_name + '_list.txt'
    #     with open(tmp_list_file, 'w') as f_list:
    #         for i in target_frame_indices:
    #             # based on sample rate, decide whether this frame is sampled
    #             line = 'file \'{}/{:04}.jpg\'\n'.format(self._image_path, i)
    #             f_list.write(line)
    #
    #     # compress the sampled image into a video
    #     frame_size = str(self._resolution[0]) + 'x' + str(self._resolution[1])
    #
    #     cmd = ['ffmpeg', '-y', '-loglevel', 'panic', '-r',
    #            str(target_frame_rate), '-f', 'concat', '-safe', '0', '-i',
    #            tmp_list_file, '-s', frame_size,
    #            '-vcodec', 'libx264', '-crf', '25', '-pix_fmt',
    #            'yuv420p', '-hide_banner', output_video_name]
    #     subprocess.run(cmd, check=True)
    #     # get the video size
    #     video_size = os.path.getsize(output_video_name)
    #     if not save_video:
    #         os.remove(output_video_name)
    #     os.remove(tmp_list_file)
    #     print('target fps={}, target resolution={}, video size={}'
    #           .format(target_frame_rate, self._resolution, video_size))
    #     print('finish generating {} and size={}'.format(
    #         output_video_name, video_size))
    #     return video_size


class MOT16Video(Video):
    """Class of MOT16Video."""

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

    @staticmethod
    def convert_grondtruth(gt_file):
        gt = pd.read_csv(gt_file)
        return None
    # TODO: finish MOT16 dataset abstraction
