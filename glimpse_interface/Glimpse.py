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
    def __init__(self, Config):
        cache_size, sample_rate, frame_diff_thres, tracking_err_thres = Config["cache_size"], Config["sample_rate"], Config["frame_diff_thres"], Config["tracking_error_thres"]
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

    def run(self, Seg, Config, Decision, Result = None):
        target_frames = []
        video, start_frame, end_frame = Seg["video"], Seg["start_frame"], Seg["end_frame"]
        cache_size, sample_rate, frame_diff_thres, tracking_err_thres = Config["cache_size"], Config["sample_rate"], Config["frame_diff_thres"], Config["tracking_error_thres"]
        self.cache_size = cache_size
        self.sample_rate = sample_rate
        self.frame_diff_thres = frame_diff_thres
        self.tracking_err_thres = tracking_err_thres
        for i in tqdm(range(start_frame, end_frame + 1, self.cache_size)):
            target_frames.append(i)
            target_frames.extend(self.selecting_frame(video, i+1,
                                             min(i + self.cache_size, end_frame), 
                                             int(self.cache_size//self.sample_rate)))
        for i in range(start_frame, end_frame + 1):
            if i not in Decision["Frame_Decision"]:
                Decision["Frame_Decision"][i] = {"skip":True}
            Decision["Frame_Decision"][i]["skip"] = True
            if i in target_frames:
                Decision["Frame_Decision"][i]["skip"] = False
        
        return Seg, Decision, Result

class Glimpse_Model(interface.Model):
    def __init__(self, Config):
        frame_diff_thres, tracking_err_thres = Config["frame_diff_thres"], Config["tracking_error_thres"]
        self.frame_diff_thres = frame_diff_thres
        self.tracking_err_thres = tracking_err_thres
        self.trackers_dict = {}

    def init_trackers(self, frame_idx, frame, boxes):
        """Return the tracked bounding boxes on input frame."""
        resolution = (frame.shape[0], frame.shape[1])
        frame_copy = cv2.resize(frame, (640, 480))
        self.trackers_dict = {}
        for box in boxes:
            xmin, ymin, xmax, ymax, t, score, obj_id = box
            tracker = cv2.TrackerKCF_create()
            # TODO: double check the definition of box input
            tracker.init(frame_copy, (xmin * 640 / resolution[0],
                                      ymin * 480 / resolution[1],
                                      (xmax - xmin) * 640 / resolution[0],
                                      (ymax - ymin) * 480 / resolution[1]))
            key = '_'.join([str(frame_idx), str(obj_id), str(t)])
            self.trackers_dict[key] = tracker

    def update_trackers(self, frame):
        """Return the tracked bounding boxes on input frame."""
        resolution = (frame.shape[0], frame.shape[1])
        frame_copy = cv2.resize(frame, (640, 480))
        start_t = time.time()
        boxes = []
        to_delete = []
        status = True
        for obj, tracker in self.trackers_dict.items():
            _, obj_id, t = obj.split('_')
            #ok, bbox = tracker.update(frame_copy)
            try:
                ok, bbox = tracker.update(frame_copy)
            except:
                status = False
            if ok:
                # tracking succeded
                x, y, w, h = bbox
                boxes.append([int(x*resolution[0]/640),
                              int(y*resolution[1]/480),
                              int((x+w)*resolution[0]/640),
                              int((y+h)*resolution[1]/480), int(float(t)),
                              1, obj_id])
            else:
                # tracking failed
                # record the trackers that need to be deleted
                status = False
                to_delete.append(obj)
        for obj in to_delete:
            self.trackers_dict.pop(obj)
        #debug_print("tracking used: {}s".format(time.time()-start_t))

        return status,boxes,0

    def run(self, Seg, Config, Decision, Results = None):
        video, start_frame, end_frame = Seg["video"], Seg["start_frame"], Seg["end_frame"]
        frame_diff_thres, tracking_err_thres = Config["frame_diff_thres"], Config["tracking_error_thres"]
        tracking_method = Config["tracking_method"]
        self.frame_diff_thres = frame_diff_thres
        self.tracking_err_thres = tracking_err_thres
        target_frames = []
        for frame_id in Decision["Frame_Decision"]:
            if not Decision["Frame_Decision"][frame_id]["skip"]:
                target_frames.append(frame_id)
        assert len(target_frames) >= 2, 'frames to detect must more than 2'
        prev_frame_gray = get_video_frame_gray(video, start_frame)
        prev_boxes = video.get_frame_detection(start_frame)
        if tracking_method == "KCF":
            self.init_trackers(start_frame, video.get_frame_image(start_frame),
                            prev_boxes)
        print(len(target_frames), end_frame - start_frame + 1)
        last_tracking_frame = copy.deepcopy(prev_frame_gray)
        last_triggered_frame = copy.deepcopy(prev_frame_gray)
        detection_results = [prev_boxes]
        trggered_frame = 0
        tracked_frame = len(target_frames)
        for i in tqdm(range(start_frame + 1, end_frame + 1)):
            frame_gray = get_video_frame_gray(video, i)
            frame_bgr = video.get_frame_image(i)
            frame_diff, _, _, _ = frame_difference(last_triggered_frame, frame_gray)
            if i not in Decision["Frame_Decision"]:
                Decision["Frame_Decision"][i] = {"skip":True}
            Decision["Frame_Decision"][i]["skip"] = True
            if frame_diff > self.frame_diff_thres:
                prev_boxes = video.get_frame_detection(i)
                last_tracking_frame = copy.deepcopy(frame_gray)
                last_triggered_frame = copy.deepcopy(frame_gray)
                Decision["Frame_Decision"][i]["skip"] = False
                trggered_frame += 1
                if tracking_method == "KCF":
                    self.init_trackers(i, video.get_frame_image(i),
                                    prev_boxes)
            elif i in target_frames:
                tracking_status = True

                if tracking_method == "KCF":
                    tracking_status,prev_boxes,err = self.update_trackers(frame_bgr)
                else:
                    tracking_status, prev_boxes, err = tracking_boxes(frame_bgr, last_tracking_frame,                                           frame_gray, i, detection_results[-1],self.tracking_err_thres, tracking_method)
                
                if not tracking_status:
                    prev_boxes = video.get_frame_detection(i)
                    if tracking_method == "KCF":
                        self.init_trackers(i, video.get_frame_image(i),
                                    prev_boxes)
                    trggered_frame += 1
                    last_triggered_frame = copy.deepcopy(frame_gray)
                    Decision["Frame_Decision"][i]["skip"] = False

                last_tracking_frame = copy.deepcopy(frame_gray)
            prev_frame_gray = copy.deepcopy(frame_gray)
            detection_results.append(prev_boxes)
        print("triggered frame {}, tracked frame {}".format(trggered_frame, tracked_frame))
        return Seg, Decision, detection_results

class Glimpse(Pipeline):
    def __init__(self, temporal_prune:Glimpse_Temporal, model_prune:Glimpse_Model):
        self.temporal_prune = temporal_prune
        self.model_prune = model_prune

    def run(self, Seg, Config, Decision = None, Results = None):
        Decision = {"Frame_Decision":{}}
        print("Selecting frames to be tracked ...")
        Seg, Decision, _ = self.temporal_prune.run(Seg, Config, Decision)
        #target_frames = [3,4]
        print("Detecting frames")
        return self.model_prune.run(Seg, Config, Decision)
    
    def evaluate(self, Seg, Config, Decision = None, Results = None):
        total_tp, total_fp, total_fn = 0,0,0
        _, decision_results, detection_results = self.run(Seg, Config)
        video, start_frame, end_frame = Seg["video"], Seg["start_frame"], Seg["end_frame"]
        for i, result in enumerate(detection_results):
            tp, fp, fn = evaluate_frame(video.get_frame_detection(start_frame + i), result)
            total_tp += tp
            total_fp += fp
            total_fn += fn
        f1_score = compute_f1(total_tp, total_fp, total_fn)
        origin_filesize = 0
        filesize = 0
        frame_num = 0
        original_frame_num = 0
        for i in range(start_frame, end_frame + 1):
            origin_filesize += video.get_frame_filesize(i)
            original_frame_num += 1
            if i in decision_results["Frame_Decision"]:
                if not decision_results["Frame_Decision"][i]["skip"]:
                    filesize += video.get_frame_filesize(i)
                    frame_num += 1
        bw = 1.0*filesize/origin_filesize
        gpu = 1.0*frame_num/original_frame_num
        return f1_score, bw, gpu

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
                   tracking_error_thresh, tracking_method):
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
    sift = cv2.xfeatures2d.SIFT_create()
    for x, y, xmax, ymax, t, score, obj_id in old_boxes:
        # mask[y:ymax, x:xmax] = 255
        corners = None
        if tracking_method == "sift":
            kps, des = sift.detectAndCompute(oldFrameGray[int(y):int(ymax), int(x):int(xmax)],None)
        #corners = cv2.goodFeaturesToTrack(oldFrameGray[int(y):int(ymax), int(x):int(xmax)],
        #                                  **feature_params)
            if len(kps) > 0:
                corners = np.zeros((len(kps),1,2), dtype=np.float32)
            for i, p in enumerate(kps):
                corners[i,0,0] = p.pt[0]
                corners[i,0,1] = p.pt[1]
        if tracking_method == "corner":
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
