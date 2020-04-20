"""Implementation of Glimpse Client."""
import copy
import http.client
# import pdb
import json
import subprocess

import cv2
import numpy as np

from videos.video import Video

BUFFER_SIZE = 1024 * 4  # 4KB


class Client(object):
    """Implementation of Glimpse Client."""

    def __init__(self, server_addr, port, video_path, config_1, config_2):
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

    def send_stream(self):
        """Send stream to server."""
        cmd = ['ffmpeg', '-i', self.video.video_path,  '-pix_fmt', 'bgr24',
               '-vcodec', 'rawvideo', '-an', '-sn', '-f', 'image2pipe', '-',
               '-hide_banner']
        pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                bufsize=BUFFER_SIZE)
        resolution = self.video.resolution

        prev_frame_gray = None
        prev_triggered_frame_gray = None
        frame_idx = 0
        while(True):
            # read the image from the video file
            raw_frame = pipe.stdout.read(resolution[0]*resolution[1]*3)
            if not raw_frame:
                # file transmitting is done
                break

            # convert read bytes to np
            frame = np.fromstring(raw_frame, dtype='uint8')
            frame = frame.reshape((resolution[1], resolution[0], 3))
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if (prev_frame_gray is None and prev_triggered_frame_gray is None)\
                    or frame_difference(prev_frame_gray, frame_gray, None,
                                        None) > self.frame_diff_thresh:
                boxes = self.send_frame(raw_frame)
                boxes = json.loads(boxes)
                prev_triggered_frame_gray = copy.deepcopy(frame_gray)
                self.init_trackers(frame_idx, frame, boxes)
            else:
                # TODO: Tracking
                boxes = self.update_trackers(frame)

            prev_frame_gray = copy.deepcopy(frame_gray)
            frame_idx += 1
            print(boxes)

        pipe.stdout.flush()

    def send_frame(self, raw_frame):
        """Send a frame to server."""
        headers = {'Content-type': 'application/octet-stream'}

        command = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
                   '-s', '1280x720',  '-pix_fmt', 'bgr24', '-i', '-',
                   '-an',  # Tells FFMPEG not to expect any audio
                   '-vcodec', 'mjpeg', '-f', 'image2', '-frames:v', '1',
                   '-qscale:v', '2', '-']
        proc = subprocess.Popen(command, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        compressed_frame = proc.communicate(raw_frame)[0]
        # print(len(raw_frame), len(compressed_frame))
        self.conn.request('POST', '/post', compressed_frame, headers)
        response = self.conn.getresponse()
        return response.read().decode()

    def init_trackers(self, frame_idx, frame, boxes):
        """Initialize trackers on Glimpose client."""
        self.trackers_dict = {}
        frame_copy = cv2.resize(frame, (640, 480))
        for box in boxes:
            xmin, ymin, xmax, ymax, t, score, obj_id = box
            tracker = cv2.TrackerKCF_create()
            # TODO: double check the definition of box input
            tracker.init(frame_copy,
                         (xmin*640/1280, ymin*480/720,
                          (xmax-xmin)*640/1280, (ymax-ymin)*480/720))
            self.trackers_dict[str(frame_idx)+'_'+str(obj_id)] = tracker

    def update_trackers(self, frame):
        """Return the tracked bounding boxes on new frame."""
        frame_copy = cv2.resize(frame, (640, 480))
        # start_t = time.time()
        boxes = []
        to_delete = []
        for obj, tracker in self.trackers_dict.items():
            obj_id = obj.split('_')[1]
            ok, bbox = tracker.update(frame_copy)
            if ok:
                # tracking succeded
                # TODO: change the box format
                x, y, w, h = bbox
                boxes.append([int(x*1280/640), int(y*720/480),
                              int((x+w)*1280/640), int((y+h)*720/480),
                              3, 1, int(obj_id)])
            else:
                # tracking failed
                # record the trackers that need to be deleted
                to_delete.append(obj)
        for obj in to_delete:
            self.trackers_dict.pop(obj)
        # debug_print("tracking used: {}s".format(time.time()-start_t))

        return boxes


def frame_difference(old_frame, new_frame, bboxes_last_triggered, bboxes,
                     thresh=35):
    """Compute the sum of pixel differences which are greater than thresh."""
    # thresh = 35 is used in Glimpse paper
    # pdb.set_trace()
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


def main():
    """Perform test."""
    client = Client('localhost', 10000, '/data/zxxia/videos/test/traffic.mp4',
                    4, 2)
    # client.connect('localhost', 10000)
    client.send_stream()


if __name__ == "__main__":
    main()
