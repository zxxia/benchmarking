"""Implementation of Glimpse Client."""
import argparse
import copy
import csv
import http.client
import json
import os
import subprocess
import time

import cv2
import numpy as np

from videos.video import Video

BUFFER_SIZE = 1024 * 4  # 4KB


class Client(object):
    """Implementation of Glimpse Client."""

    def __init__(self, server_addr, port, video_path, config_1, config_2=None):
        """Client initialization.

        Args
            video_path(str): path to the source video.
            chunk_size(int): number of frames to encode per chunk.

        """
        self.video = Video(video_path)
        print('connect to ({}, {})'.format(server_addr, port))
        self.conn = http.client.HTTPConnection(server_addr, port)

        self.frame_diff_thresh = self.video.resolution[0] * \
            self.video.resolution[1]/config_1

        self.trackers_dict = None  # trackers
        self.detections = {}
        self.profile = {}
        self.bytes_sent = {}

    def send_stream(self):
        """Send stream to server."""
        cmd = ['ffmpeg', '-i', self.video.video_path,  '-pix_fmt', 'bgr24',
               '-vcodec', 'rawvideo', '-an', '-sn', '-f', 'image2pipe', '-',
               '-hide_banner']
        try:
            pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    bufsize=BUFFER_SIZE)
            resolution = self.video.resolution

            prev_frame_gray = None
            prev_triggered_frame_gray = None
            frame_idx = 0
            while(True):
                # read the image from the video file
                raw_frame = pipe.stdout.read(resolution[0]*resolution[1]*3)
                if not raw_frame:  # file transmitting is done
                    break

                # convert read bytes to np
                frame = np.fromstring(raw_frame, dtype='uint8')
                frame = frame.reshape((resolution[1], resolution[0], 3))
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if (prev_frame_gray is None and
                    prev_triggered_frame_gray is None)\
                        or frame_difference(prev_frame_gray, frame_gray, None,
                                            None) > self.frame_diff_thresh:
                    # TODO: maintain an active cache and implement the frame
                    # picking logic described in the paper.
                    boxes, t_used, bytes_sent = self.send_frame(raw_frame)
                    t_used = float(t_used)
                    prev_triggered_frame_gray = copy.deepcopy(frame_gray)
                    self.init_trackers(frame_idx, frame, boxes)
                else:
                    # Tracking
                    boxes, t_used = self.update_trackers(frame)
                    t_used = 0  # gpu time used is 0
                    bytes_sent = 0

                prev_frame_gray = copy.deepcopy(frame_gray)
                self.detections[frame_idx] = boxes
                self.profile[frame_idx] = t_used
                self.bytes_sent[frame_idx] = bytes_sent
                frame_idx += 1
                print('frame {}: {}'.format(frame_idx, len(boxes)))
                # print(boxes)

                # color = yellow = (0, 255, 255)
                # for box in boxes:
                #     [xmin, ymin, xmax, ymax] = box[:4]
                #     cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax),
                # int(ymax)), color, 1)
                # cv2.imshow(str(frame_idx), frame)
                # cv2.moveWindow(str(frame_idx), 0, 0)
                # if cv2.waitKey(0) & 0xFF == ord('q'):
                #     cv2.destroyAllWindows()
        finally:
            pipe.stdout.flush()

    def send_frame(self, raw_frame):
        """Send a frame to server."""
        resolution = self.video.resolution
        headers = {'Content-type': 'application/octet-stream'}

        command = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
                   '-s', '{}x{}'.format(resolution[0], resolution[1]),
                   '-pix_fmt', 'bgr24', '-i', '-',
                   '-an',  # Tells FFMPEG not to expect any audio
                   '-vcodec', 'mjpeg', '-f', 'image2', '-frames:v', '1',
                   '-qscale:v', '2', '-', '-hide_banner']
        proc = subprocess.Popen(command, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        compressed_frame = proc.communicate(raw_frame)[0]
        # print(len(raw_frame), len(compressed_frame))
        self.conn.request('POST', '/post', compressed_frame, headers)
        response = self.conn.getresponse()
        boxes, t_used = json.loads(response.read().decode())
        return boxes, t_used, len(compressed_frame)

    def init_trackers(self, frame_idx, frame, boxes):
        """Initialize trackers on Glimpose client."""
        resolution = self.video.resolution
        self.trackers_dict = {}
        frame_copy = cv2.resize(frame, (640, 480))
        for obj_id, box in enumerate(boxes):
            xmin, ymin, xmax, ymax, t, score = box
            tracker = cv2.TrackerKCF_create()
            tracker.init(frame_copy,
                         (xmin*640/resolution[0], ymin*480/resolution[1],
                          (xmax-xmin)*640/resolution[0],
                          (ymax-ymin)*480/resolution[1]))
            self.trackers_dict[f'{frame_idx}_{obj_id}_{t}'] = tracker

    def update_trackers(self, frame):
        """Return the tracked bounding boxes on new frame."""
        resolution = self.video.resolution
        frame_copy = cv2.resize(frame, (640, 480))
        start_t = time.time()
        boxes = []
        to_delete = []
        for obj, tracker in self.trackers_dict.items():
            # obj_id = obj.split('_')[1]
            t = obj.split('_')[-1]
            ok, bbox = tracker.update(frame_copy)
            if ok:
                # tracking succeded
                x, y, w, h = bbox
                boxes.append([x*resolution[0]/640, y*resolution[1]/480,
                              (x+w)*resolution[0]/640, (y+h)*resolution[1]/480,
                              int(t), 1])
            else:
                # tracking failed
                # record the trackers that need to be deleted
                to_delete.append(obj)
        for obj in to_delete:
            self.trackers_dict.pop(obj)
        t_used = time.time() - start_t

        return boxes, t_used

    def dump_detections(self, filename):
        """Dump detections to disk."""
        with open(filename, 'w', 1) as f:
            writer = csv.writer(f)
            header = ['frame id', 'xmin', 'ymin', 'xmax', 'ymax',
                      'class', 'score']
            writer.writerow(header)
            for frame_id in sorted(self.detections):
                boxes = self.detections[frame_id]
                if not boxes:
                    writer.writerow([frame_id, '', '', '', '', '', ''])
                for box in boxes:
                    xmin, ymin, xmax, ymax, obj_type, score = box
                    writer.writerow(
                        [frame_id, xmin, ymin, xmax, ymax, obj_type, score])

    def dump_profile(self, filename):
        """Dump profile to disk."""
        with open(filename, 'w', 1) as f:
            writer = csv.writer(f)
            writer.writerow(['frame id', 'gpu time used(s)', 'bytes'])
            for frame_id in sorted(self.profile):
                t_used = self.profile[frame_id]
                bytes_sent = self.bytes_sent[frame_id]
                writer.writerow([frame_id, t_used, bytes_sent])


def frame_difference(old_frame, new_frame, bboxes_last_triggered, bboxes,
                     thresh=35):
    """Compute the sum of pixel differences which are greater than thresh."""
    # thresh = 35 is used in Glimpse paper
    # start_t = time.time()
    diff = np.absolute(new_frame.astype(int) - old_frame.astype(int))
    mask = np.greater(diff, thresh)
    pix_change = np.sum(mask)
    # time_elapsed = time.time() - start_t
    # print('frame difference used: {}'.format(time_eplased*1000))
    # pix_change_obj = 0
    # obj_region = np.zeros_like(new_frame)
    # for box in bboxes_last_triggered:
    #     xmin, ymin, xmax, ymax = box[:4]
    #     obj_region[ymin:ymax, xmin:xmax] = 1
    # for box in bboxes:
    #     xmin, ymin, xmax, ymax = box[:4]
    #     obj_region[ymin:ymax, xmin:xmax] = 1
    # pix_change_obj += np.sum(mask * obj_region)
    # pix_change_bg = pix_change - pix_change_obj

    # cv2.imshow('frame diff', np.repeat(
    #     mask[:, :, np.newaxis], 3, axis=2).astype(np.uint8))
    # cv2.moveWindow('frame diff', 1280, 0)
    # if cv2.waitKey(0) & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()
    # cv2.destroyWindow('frame diff')

    return pix_change  # , pix_change_obj, pix_change_bg, time_elapsed


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="Glimpse Client.")
    parser.add_argument("--hostname", type=str, required=True,
                        help="Hostname to connect to.")
    parser.add_argument("--port", type=int, required=True,
                        help="Port to connect to.")
    parser.add_argument("--video_path", type=str, required=True,
                        help="Path to video file.")
    parser.add_argument("--config1", type=int, default=10,
                        help="frame differece thresh = pixel number/config1")
    parser.add_argument("--output_path", type=str, required=True,
                        help="output path where output files will be saved to")
    args = parser.parse_args()
    return args


def main():
    """Perform test."""
    args = parse_args()
    client = Client(args.hostname, args.port, args.video_path, args.config1)
    client.send_stream()
    client.dump_detections(os.path.join(args.output_path, 'dets.csv'))
    client.dump_profile(os.path.join(args.output_path, 'profile.csv'))


if __name__ == "__main__":
    main()
