"""Definition of YoutubeVideo class."""
import copy
import os
import subprocess

import cv2

from constants import CAMERA_TYPES, RESOL_DICT, COCOLabels
from object_detection.infer import load_object_detection_results
from utils.utils import filter_video_detections, load_full_model_detection
# remove_overlappings)
from videos.video import Video


class YoutubeVideo(Video):
    """Class of YoutubeVideo."""

    def __init__(self, root, name, resolution_name,
                 model='faster_rcnn_resnet101', qp=23, filter_flag=True,
                 merge_label_flag=False,
                 classes_interested={COCOLabels.CAR.value,
                                     COCOLabels.BUS.value,
                                     COCOLabels.TRUCK.value}, cropped=False):
        """Youtube Video Constructor."""
        resolution = RESOL_DICT[resolution_name]
        video_path = os.path.join(root, name+'.mp4')
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
        if isinstance(video_path, str) and os.path.exists(video_path):
            vid = cv2.VideoCapture(video_path)
            fps = int(round(vid.get(cv2.CAP_PROP_FPS)))
            vid.release()
        else:
            fps = 30

        if isinstance(detection_file, str) and os.path.exists(detection_file):
            # dets = load_full_model_detection(detection_file)
            print('loading {}...'.format(detection_file))
            dets = load_object_detection_results(detection_file)
            dets_nofilter = copy.deepcopy(dets)

            # TODO: handle overlapping boxes
            if name in CAMERA_TYPES['static']:
                camera_type = 'static'
                if filter_flag:
                    # TODO: change label loading logic
                    dets, dropped_dets = filter_video_detections(
                        dets,
                        target_types=classes_interested,
                        score_range=(0.3, 1.0),
                        width_range=(resolution[0] // 20, resolution[0]/2),
                        height_range=(resolution[1] // 20, resolution[1]/2)
                    )
                    #     dets[frame_idx] = remove_overlappings(boxes, 0.3)
                else:
                    dropped_dets = None

            elif name in CAMERA_TYPES['moving']:
                camera_type = 'moving'
                if filter_flag:
                    dets, dropped_dets = filter_video_detections(
                        dets,
                        target_types=classes_interested,
                        height_range=(resolution[1] // 20, resolution[1]))
                    #     dets[frame_idx] = remove_overlappings(boxes, 0.3)
                else:
                    dropped_dets = None
            else:
                if filter_flag:
                    # TODO: change label loading logic
                    dets, dropped_dets = filter_video_detections(
                        dets,
                        target_types=classes_interested,
                        score_range=(0.3, 1.0),
                        # width_range=(resolution[0] // 20, resolution[0]/2),
                        height_range=(resolution[1] // 20, resolution[1])
                    )
                    #     dets[frame_idx] = remove_overlappings(boxes, 0.3)
                else:
                    dropped_dets = None
                camera_type = None
                # dropped_dets = None

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
                    # dets[frame_idx] = remove_overlappings(tmp_boxes, 0.3)
                    dets[frame_idx] = tmp_boxes

            if merge_label_flag:
                for frame_idx, boxes in dets.items():
                    for box_idx, _ in enumerate(boxes):
                        # Merge all cars and trucks into cars
                        dets[frame_idx][box_idx][4] = COCOLabels.CAR.value
        else:
            print(detection_file, "does not exist.")
            raise NotImplementedError

        super().__init__(name, fps, resolution, dets, dets_nofilter,
                         image_path, camera_type, model, dropped_dets)

    def get_frame_image_name(self, frame_index):
        """Return the image file name at frame index."""
        img_file = os.path.join(
            self._image_path, '{:06d}.jpg'.format(frame_index))
        return img_file

    # a better way to create a video
    def encode_ffmpeg(self, video, frame_range, every_n_frame, output_path):
        """Create a video using ffmepg."""
        # TODO: add path to original video
        output_video_name = os.path.join(output_path, "{}.mp4".format(video))
        frame_size = str(self._resolution[0]) + 'x' + str(self._resolution[1])
        cmd = "ffmpeg -y -i {} -an -vf " \
            "select=between(n\,{}\,{})*not(mod(n\,{})),setpts=PTS-STARTPTS " \
            "-vsync vfr -s {} -crf {} {} -hide_banner".format(
                video, frame_range[0], frame_range[1], every_n_frame,
                frame_size, self._quality_level, output_video_name)
        print(cmd)
        subprocess.run(cmd.split(' '), check=True)
