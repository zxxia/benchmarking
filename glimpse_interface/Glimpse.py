"""Offline Glimpse Implementation."""
import copy
import csv
import os
import pdb
import time
from collections import defaultdict

import cv2
import numpy as np

from evaluation.f1 import compute_f1, evaluate_frame
from utils.utils import interpolation
import interface
from pipeline import Pipeline
from tqdm import tqdm
import glimpse

def debug_print(msg):
    print(msg)

def get_video_frame_gray(video, frame_id):
        frame_gray = cv2.cvtColor(video.get_frame_image(frame_id),
                                  cv2.COLOR_BGR2GRAY)
        assert frame_gray is not None, 'cannot read image {}'.format(
            frame_id)
        return frame_gray

class Glimpse_Temporal(interface.Temporal):
    def __init__(self, cache_size, sample_rate, frame_diff_thres, tracking_err_thres):
        self.cache_size = cache_size
        self.sample_rate = sample_rate
        self.frame_diff_thres = frame_diff_thres
        self.tracking_err_thres = tracking_err_thres

    def selecting_frame(self, video, frame_start, frame_end, num_frame):
        if frame_start >= frame_end:
            return []
        frame_gray = get_video_frame_gray(video, frame_start)
        prev_frame_gray = frame_gray.copy()
        frame_diffs = []
        target_frames = []
        for i in range(frame_start + 1, frame_end + 1):
            frame_gray = get_video_frame_gray(video, i)
            frame_diff, _, _, _ = frame_difference(prev_frame_gray, frame_gray)
            frame_diffs.append(frame_diff)
            prev_frame_gray = frame_gray.copy()
        min_diff = 0
        max_diff = video.resolution[0]*video.resolution[1]
        selected_frames = []
        while(min_diff < max_diff - 1):
            mid_diff = (min_diff + max_diff)//2
            selected_frames = []
            selected_nb = 0
            total_diff = 0
            for i, diff in enumerate(frame_diffs):
                total_diff += diff
                if total_diff > mid_diff:
                    selected_frames.append(frame_start + i + 1)
                    selected_nb += 1
                    total_diff = 0
            if selected_nb > num_frame:
                min_diff = mid_diff + 1
            else:
                max_diff = mid_diff
        total_diff = 0
        for i, diff in enumerate(frame_diffs):
            total_diff += diff
            if total_diff > max_diff:
                target_frames.append(frame_start + i + 1)
                total_diff = 0
        return target_frames

    def run(self, video, start_frame, end_frame):
        target_frames = []
        for i in tqdm(range(start_frame, end_frame + 1, self.cache_size)):
            target_frames.append(i)
            target_frames.extend(self.selecting_frame(video, i+1,
                                             min(i + self.cache_size, end_frame), 
                                             int(self.cache_size//self.sample_rate)))
        return target_frames

class Glimpse_Model(interface.Model):
    def __init__(self, frame_diff_thres, tracking_err_thres):
        self.frame_diff_thres = frame_diff_thres
        self.tracking_err_thres = tracking_err_thres

    def run(self, video, start_frame, end_frame, target_frames):
        assert len(target_frames) >= 2, 'frames to detect must more than 2'
        prev_frame_gray = get_video_frame_gray(video, start_frame)
        prev_boxes = video.get_frame_detection(start_frame)
        print(len(target_frames), end_frame - start_frame + 1)
        last_tracking_frame = copy.deepcopy(prev_frame_gray)
        detection_results = [prev_boxes]
        trggered_frame = 0
        tracked_frame = len(target_frames)
        for i in tqdm(range(start_frame + 1, end_frame + 1)):
            frame_gray = get_video_frame_gray(video, i)
            frame_bgr = video.get_frame_image(i)
            frame_diff, _, _, _ = frame_difference(prev_frame_gray, frame_gray)
            assert i in target_frames
            if frame_diff > self.frame_diff_thres:
                prev_boxes = video.get_frame_detection(i)
                last_tracking_frame = copy.deepcopy(frame_gray)
                trggered_frame += 1
            elif i in target_frames:
                tracking_status, prev_boxes, err = tracking_boxes(frame_bgr, last_tracking_frame,                                           frame_gray, i, detection_results[-1],self.tracking_err_thres)
                if not tracking_status:
                    prev_boxes = video.get_frame_detection(i)
                    trggered_frame += 1
                last_tracking_frame = copy.deepcopy(frame_gray)
            prev_frame_gray = copy.deepcopy(frame_gray)
            detection_results.append(prev_boxes)
        print("triggered frame {}, tracked frame {}".format(trggered_frame, tracked_frame))
        return detection_results, trggered_frame

class Glimpse(Pipeline):
    def __init__(self, temporal_prune:Glimpse_Temporal, model_prune:Glimpse_Model):
        self.temporal_prune = temporal_prune
        self.model_prune = model_prune

    def run(self, video, start_frame, end_frame, output_file, vis_video=False, output_video=None):
        print("Selecting frames to be tracked ...")
        target_frames = self.temporal_prune.run(video, start_frame, end_frame)
        #target_frames = [3,4]
        frame_results = None
        with open(output_file, 'w', 1) as f_out:
            writer = csv.writer(f_out)
            writer.writerow(['frame id', 'xmin', 'ymin', 'xmax', 
                            'ymax', 'class' ,' score', 'object id'])
            print("Start detecting")
            frame_results, triggered_frame = self.model_prune.run(video, start_frame, end_frame, target_frames)
            yellow = (0, 255, 255)
            videoWriter = None
            if vis_video and output_video is not None:
                fps = 30
                size = (video.resolution[0],video.resolution[1])
                fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
                videoWriter = cv2.VideoWriter(output_video, fourcc, fps, size) 
            print("saving results ...") 
            total_tp, total_fp, total_fn = 0,0,0
            for i, result in enumerate(frame_results):
                tp, fp, fn = evaluate_frame(video.get_frame_detection(start_frame + i), result)
                total_tp += tp
                total_fp += fp
                total_fn += fn
                if vis_video and output_video is not None:
                    frame = video.get_frame_image(start_frame + i)
                for box in result:
                    writer.writerow([i+1] + box)
                    if vis_video and output_video is not None:
                        cv2.rectangle(frame, (int(box[0]),int(box[1])), 
                                    (int(box[2]),int(box[3])), yellow, 2)
                if vis_video and output_video is not None:
                    videoWriter.write(frame)
            if vis_video and output_video is not None:
                videoWriter.release()
            f1_score = compute_f1(total_tp, total_fp, total_fn)
        return frame_results, triggered_frame, f1_score

def frame_difference(old_frame, new_frame, bboxes_last_triggered = None, bboxes = None,
                     thresh=35):
    """Compute the sum of pixel differences which are greater than thresh."""
    # thresh = 35 is used in Glimpse paper
    # pdb.set_trace()
    start_t = time.time()
    diff = np.absolute(new_frame.astype(int) - old_frame.astype(int))
    mask = np.greater(diff, thresh)
    pix_change = np.sum(mask)
    time_elapsed = time.time() - start_t
    #debug_print('frame difference used: {}'.format(time_elapsed*1000))
    pix_change_obj = 0
    # obj_region = np.zeros_like(new_frame)
    # for box in bboxes_last_triggered:
    #     xmin, ymin, xmax, ymax = box[:4]
    #     obj_region[int(ymin):int(ymax), int(xmin):int(xmax)] = 1
    # for box in bboxes:
    #     xmin, ymin, xmax, ymax = box[:4]
    #     obj_region[int(ymin):int(ymax), int(xmin):int(xmax)] = 1
    # pix_change_obj += np.sum(mask * obj_region)
    pix_change_bg = pix_change - pix_change_obj

    # cv2.imshow('frame diff', np.repeat(
    #     mask[:, :, np.newaxis], 3, axis=2).astype(np.uint8))
    # cv2.moveWindow('frame diff', 1280, 0)
    # if cv2.waitKey(0) & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()
    # cv2.destroyWindow('frame diff')

    return pix_change, pix_change_obj, pix_change_bg, time_elapsed


def tracking_boxes(vis, oldFrameGray, newFrameGray, new_frame_id, old_boxes,
                   tracking_error_thresh):
    """
    Tracking the bboxes between frames via optical flow.

    Arg
        vis(numpy array): an BGR image which helps visualization
        oldFrameGray(numpy array): a grayscale image of previous frame
        newFrameGray(numpy array): a grayscale image of current frame
        new_frame_id(int): frame index
        old_boxes(list): a list of boxes in previous frame
        tracking_error_thresh(float): tracking error threshold
    Return
        tracking status(boolean) - tracking success or failure
        new bboxes tracked by optical flow
    """
    # define colors for visualization
    yellow = (0, 255, 255)
    black = (0, 0, 0)

    # define optical flow parameters
    lk_params = dict(winSize=(15, 15), maxLevel=2,  # 5,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                               10, 0.03))

    # define good feature compuration parameters
    feature_params = dict(maxCorners=50, qualityLevel=0.01,
                          minDistance=7, blockSize=7)

    # mask = np.zeros_like(oldFrameGray)
    start_t = time.time()
    old_corners = []
    for x, y, xmax, ymax, t, score, obj_id in old_boxes:
        # mask[y:ymax, x:xmax] = 255
        corners = cv2.goodFeaturesToTrack(oldFrameGray[int(y):int(ymax), int(x):int(xmax)],
                                          **feature_params)
        if corners is not None:
            corners[:, 0, 0] = corners[:, 0, 0] + x
            corners[:, 0, 1] = corners[:, 0, 1] + y
            old_corners.append(corners)
    # print('compute feature {}seconds'.format(time.time() - start_t))
    if not old_corners:
        # cannot find available corners and treat as objects disappears
        return True, [], 0
    else:
        old_corners = np.concatenate(old_corners)

    # old_corners = cv2.goodFeaturesToTrack(oldFrameGray, mask=mask,
    #                                       **feature_params)
    # old_corners = cv2.goodFeaturesToTrack(oldFrameGray, 26, 0.01, 7,
    #                                       mask=mask)
    new_corners, st, err = cv2.calcOpticalFlowPyrLK(oldFrameGray, newFrameGray,
                                                    old_corners, None,
                                                    **lk_params)
    # old_corners_r, st, err = cv2.calcOpticalFlowPyrLK(newFrameGray,
    #                                                   oldFrameGray,
    #                                                   old_corners, None,
    #                                                   **lk_params)
    # d = abs(old_corners-old_corners_r).reshape(-1, 2).max(-1)
    # good = d < 1
    # new_corners_copy = new_corners.copy()
    # pdb.set_trace()
    new_corners = new_corners[st == 1].reshape(-1, 1, 2)
    old_corners = old_corners[st == 1].reshape(-1, 1, 2)
    # new_corners = new_corners[good]
    # old_corners = old_corners[good]

    for new_c, old_c in zip(new_corners, old_corners):
        # new corners in yellow circles
        cv2.circle(vis, (new_c[0][0], new_c[0][1]), 5, yellow, -1)
        # old corners in black circles
        cv2.circle(vis, (old_c[0][0], old_c[0][1]), 5, black, -1)

    new_boxes = []

    for x, y, xmax, ymax, t, score, obj_id in old_boxes:
        indices = []
        for idx, (old_c, new_c) in enumerate(zip(old_corners, new_corners)):
            if old_c[0][0] >= x and old_c[0][0] <= xmax and \
               old_c[0][1] >= y and old_c[0][1] <= ymax:
                indices.append(idx)
        if not indices:
            debug_print('frame {}: object {} disappear'.format(new_frame_id,
                                                               obj_id))
            continue
        indices = np.array(indices)

        # checking tracking error threshold condition
        displacement_vectors = []
        dist_list = []
        for old_corner, new_corner in zip(old_corners[indices],
                                          new_corners[indices]):
            dist_list.append(np.linalg.norm(new_corner-old_corner))
            displacement_vectors.append(new_corner-old_corner)
        tracking_err = np.std(dist_list)
        # print('tracking error:', tracking_err)
        if tracking_err > tracking_error_thresh:
            # tracking failure, this is a trigger frame
            debug_print('frame {}: '
                        'object {} std {} > tracking error thresh {}, '
                        'tracking fails'.format(new_frame_id, obj_id,
                                                np.std(dist_list),
                                                tracking_error_thresh))
            return False, [], tracking_err

        # update bouding box translational movement and uniform scaling
        # print('corner number:', old_corners[indices].shape)
        affine_trans_mat, inliers = cv2.estimateAffinePartial2D(
            old_corners[indices], new_corners[indices])
        if affine_trans_mat is None or np.isnan(affine_trans_mat).any():
            # the bbox is too small and not enough good features obtained to
            # compute reliable affine transformation matrix.
            # consider the object disappeared
            continue

        assert affine_trans_mat.shape == (2, 3)
        # print('old box:', x, y, xmax, ymax)
        # print(affine_trans_mat)
        scaling = np.linalg.norm(affine_trans_mat[:, 0])
        translation = affine_trans_mat[:, 2]
        new_x = int(np.round(scaling * x + translation[0]))
        new_y = int(np.round(scaling * y + translation[1]))
        new_xmax = int(np.round(scaling * xmax + translation[0]))
        new_ymax = int(np.round(scaling * ymax + translation[1]))
        # print('new box:', new_x, new_y, new_xmax, new_ymax)
        if new_x >= vis.shape[1] or new_xmax <= 0:
            # object disappears from the right/left of the screen
            continue
        if new_y >= vis.shape[0] or new_ymax <= 0:
            # object disappears from the bottom/top of the screen
            continue

        # The bbox are partially visible in the screen
        if new_x < 0:
            new_x = 0
        if new_xmax > vis.shape[1]:
            new_xmax = vis.shape[1]
        if new_y < 0:
            new_y = 0
        if new_ymax > vis.shape[0]:
            new_ymax = vis.shape[0]
        assert 0 <= new_x <= vis.shape[1], "new_x {} is out of [0, {}]".format(
            new_x, vis.shape[1])
        assert 0 <= new_xmax <= vis.shape[1], "new_xmax {} is out of [0, {}]"\
            .format(new_xmax, vis.shape[1])
        assert 0 <= new_y <= vis.shape[0], "new_y {} is out of [0, {}]".format(
            new_y, vis.shape[0])
        assert 0 <= new_ymax <= vis.shape[0], "new_ymax {} is out of [0, {}]"\
            .format(new_ymax, vis.shape[0])
        # pdb.set_trace()
        new_boxes.append([new_x, new_y, new_xmax, new_ymax, t, score, obj_id])
        # cv2.rectangle(vis, (x, y), (xmax, ymax), black, 2)
        # cv2.rectangle(vis, (new_x, new_y), (new_xmax, new_ymax), yellow, 2)

    # img_title = 'frame {}'.format(new_frame_id)
    # cv2.imshow(img_title, vis)
    # cv2.moveWindow(img_title, 0, 0)
    # if cv2.waitKey(0) & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()
    # else:
    #     cv2.destroyWindow(img_title)
    return True, new_boxes, 0


def eval_pipeline_accuracy(frame_start, frame_end,
                           gt_annot, dt_glimpse, iou_thresh=0.5):
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    gt_cn = 0
    dt_cn = 0
    for i in range(frame_start, frame_end + 1):
        gt_boxes_final = gt_annot[i].copy()
        dt_boxes_final = dt_glimpse[i].copy()
        gt_cn += len(gt_boxes_final)
        dt_cn += len(dt_boxes_final)
        tp[i], fp[i], fn[i] = evaluate_frame(gt_boxes_final, dt_boxes_final,
                                             iou_thresh)

    tp_total = sum(tp.values())
    fn_total = sum(fn.values())
    fp_total = sum(fp.values())

    return compute_f1(tp_total, fp_total, fn_total)


def object_appearance(start, end, gt):
    """Take start frame, end frame, and groundtruth.

    Return
        object to frame range (dict)
        frame id to new object id (dict)

    """
    obj_to_frame_range = dict()
    frame_to_new_obj = dict()
    for frame_id in range(int(start), int(end)+1):
        if frame_id not in gt:
            continue
        boxes = gt[frame_id]
        for box in boxes:
            try:
                obj_id = int(box[-1])
            except ValueError:
                obj_id = box[-1]

            if obj_id in obj_to_frame_range:
                start, end = obj_to_frame_range[obj_id]
                obj_to_frame_range[obj_id][0] = min(int(frame_id), start)
                obj_to_frame_range[obj_id][1] = max(int(frame_id), end)
            else:
                obj_to_frame_range[obj_id] = [int(frame_id), int(frame_id)]

    for obj_id in obj_to_frame_range:
        if obj_to_frame_range[obj_id][0] in frame_to_new_obj:
            frame_to_new_obj[obj_to_frame_range[obj_id][0]].append(obj_id)
        else:
            frame_to_new_obj[obj_to_frame_range[obj_id][0]] = [obj_id]

    return obj_to_frame_range, frame_to_new_obj


def compute_target_frame_rate(frame_rate_list, f1_list, target_f1=0.9):
    """Compute target frame rate when target f1 is achieved."""
    index = frame_rate_list.index(max(frame_rate_list))
    f1_list_normalized = [x/f1_list[index] for x in f1_list]
    result = [(y, x) for x, y in sorted(zip(f1_list_normalized,
                                            frame_rate_list))]
    # print(list(zip(frame_rate_list,f1_list_normalized)))
    frame_rate_list_sorted = [x for (x, _) in result]
    f1_list_sorted = [y for (_, y) in result]
    index = next(x[0] for x in enumerate(f1_list_sorted) if x[1] > target_f1)
    if index == 0:
        target_frame_rate = frame_rate_list_sorted[0]
        return target_frame_rate, -1, f1_list_sorted[index], -1,\
            frame_rate_list_sorted[index]
    else:
        point_a = (f1_list_sorted[index-1], frame_rate_list_sorted[index-1])
        point_b = (f1_list_sorted[index], frame_rate_list_sorted[index])

        target_frame_rate = interpolation(point_a, point_b, target_f1)
        return target_frame_rate, f1_list_sorted[index - 1], \
            f1_list_sorted[index], frame_rate_list_sorted[index-1], \
            frame_rate_list_sorted[index]
