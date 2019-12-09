"""Glimpse Definition."""
import csv
from collections import defaultdict
import numpy as np
import cv2
from glimpse.glimpse import compute_target_frame_rate
from utils.model_utils import eval_single_image, filter_video_detections
from utils.utils import compute_f1
from constants import COCOLabels

DEBUG = False
# DEBUG = True


def debug_print(msg):
    """Debug print."""
    if DEBUG:
        print(msg)


class Glimpse():
    """Glimpse Pipeline."""

    def __init__(self, para1_list, para2_list, profile_log, mode='default',
                 target_f1=0.9):
        """Load the configs.

        Args
            para1_list: a list of para1 used to compute frame difference thresh
            para2_list: a list of para2 used to compute tracking error thresh
            profile_log: profile log name
            mode: 'default': glimpse default tracking
                  'perfect tracking': perfect tracking by tracking boxes from
                                      last frame. boxes are from ground truth
                  'frame select': using bboxes from previously triggered frame

        """
        self.para1_list = para1_list
        self.para2_list = para2_list
        self.target_f1 = target_f1
        self.writer = csv.writer(open(profile_log, 'w', 1))
        self.writer.writerow(['video chunk', 'para1', 'para2', 'f1',
                              'frame rate,' 'ideal frame rate', 'trigger f1'])
        assert mode == 'default' or mode == 'frame select' or \
            mode == 'perfect tracking', 'wrong mode specified'
        self.mode = mode

    def profile(self, clip, video, profile_start, profile_end):
        """Profile video from profile start to profile end."""
        f1_diff = 1
        # the minimum f1 score which is greater than
        # or equal to target f1(e.g. 0.9)
        fps = video.frame_rate
        resolution = video.resolution
        best_para1 = -1
        best_para2 = -1

        test_f1_list = []
        test_fps_list = []
        for para1 in self.para1_list:
            for para2 in self.para2_list:
                # larger para1, smaller thresh, easier to be triggered
                frame_diff_th = resolution[0]*resolution[1]/para1
                tracking_error_th = para2
                # images start from index 1
                ideal_triggered_frame, f1, trigger_f1, pix_change_obj, \
                    pix_change_bg, frames_triggered = \
                    self.pipeline(video, profile_start, profile_end,
                                  frame_diff_th, tracking_error_th,
                                  view=False, mask_flag=True)
                current_fps = len(frames_triggered) / \
                    (profile_end - profile_start) * fps
                ideal_fps = ideal_triggered_frame / \
                    (profile_end - profile_start) * fps
                print('para1={}, para2={}, '
                      'Profiled f1={}, Profiled perf={}, Ideal perf={}'
                      .format(para1, para2, f1, current_fps/fps,
                              ideal_fps / fps))
                self.writer.writerow([clip, para1, para2,
                                      f1, current_fps/fps,
                                      ideal_fps/fps,
                                      trigger_f1,
                                      pix_change_obj,
                                      pix_change_bg])
                test_f1_list.append(f1)
                test_fps_list.append(current_fps)

                if abs(f1 - self.target_f1) < f1_diff:
                    f1_diff = abs(f1-self.target_f1)
                    # record the best config
                    best_para1 = para1
                    best_para2 = para2

        test_f1_list.append(1.0)
        test_fps_list.append(fps)
        final_fps, f1_left, f1_right, fps_left, fps_right =\
            compute_target_frame_rate(test_fps_list, test_f1_list)

        print("best_para1={}, best_para2={}".format(best_para1, best_para2))
        return best_para1, best_para2

    def evaluate(self, video, test_start, test_end, para1, para2):
        """Evaluate a video from test start to test end."""
        resolution = video.resolution
        frame_diff_th = resolution[0]*resolution[1]/para1
        tracking_error_th = para2
        ideal_triggered_frame, f1, trigger_f1, pix_change_obj, pix_change_bg, \
            frames_triggered = self.pipeline(video, test_start, test_end,
                                             frame_diff_th, tracking_error_th,
                                             view=False, mask_flag=True)
        return ideal_triggered_frame, f1, trigger_f1, \
            pix_change_obj, pix_change_bg, frames_triggered

    def pipeline(self, video, frame_start, frame_end,
                 frame_difference_thresh, tracking_error_thresh,
                 view=False, mask_flag=False):
        """Glimpse pipeline.

        Args
            video - a video object
            frame_start - start frame index
            frame_end - end frame index
            frame_difference_thresh - frame difference threshold
            tracking_error_thresh - tracking error threshold
            view - visulize frames when True
            mask_flag - masked out uninterested objects when computing frame
                        differences if True

        Return
            ideal_nb_triggered - ideally number of frames triggered
            f1 - F1 score of object detection on input video
            trigger_f1 - F1 score of triggering (measuring triggering accuracy)
            avg_pix_change_obj - average pixel change on object
            avg_pix_change_bg - average pixel change on background
            frames_triggered - indices of frames triggered

        """
        resolution = video.resolution

        # get filtered groudtruth
        if video.video_type == 'static':
            filtered_gt = filter_video_detections(
                video.get_video_detection(),
                target_types={COCOLabels.CAR.value,
                              COCOLabels.BUS.value,
                              COCOLabels.TRUCK.value},
                width_range=(0, resolution[0]/2),
                height_range=(0, resolution[1] / 2))
        elif video.video_type == 'moving':
            filtered_gt = filter_video_detections(
                video.get_video_detection(),
                target_types={COCOLabels.CAR.value,
                              COCOLabels.BUS.value,
                              COCOLabels.TRUCK.value},
                height_range=(resolution[1] // 20, resolution[1]))
        obj_to_frame_range, frame_to_new_obj = object_appearance(frame_start,
                                                                 frame_end,
                                                                 filtered_gt)
        ideally_triggered_frames = set()
        for obj_id in obj_to_frame_range:
            frame_range = obj_to_frame_range[obj_id]
            ideally_triggered_frames.add(frame_range[0])
        ideal_nb_triggered = len(ideally_triggered_frames)

        dt_glimpse = defaultdict(list)
        frames_triggered = set()
        # start from the first frame
        # The first frame has to be sent to server.
        frame_gray = video.get_frame_image(frame_start, True)
        assert frame_gray is not None, 'cannot read image {}'.format(
            frame_start)
        lastTriggeredFrameGray = frame_gray.copy()
        prev_frame_gray = frame_gray.copy()
        last_triggered_frame_idx = frame_start

        # get detection from server
        dt_glimpse[frame_start] = filtered_gt[frame_start]
        # video.get_frame_detection(frame_start)
        # if detection is obtained from server, set frame_flag to 1
        pix_change = np.zeros_like(frame_gray)
        # if view:
        #     view = visualize(oldFrameGray, pix_change, gt, dt_glimpse,
        # frame_start,
        #                      frame_to_new_obj)
        frames_triggered.add(frame_start)
        tp = 0

        pix_change_obj_list = list()
        pix_change_bg_list = list()

        # run the pipeline for the rest of the frames
        for i in range(frame_start + 1, frame_end + 1):
            frame_gray = video.get_frame_image(i, True)
            assert frame_gray is not None, 'cannot read {}'.format(i)

            pix_change = np.zeros_like(frame_gray)
            # mask out bboxes
            if mask_flag:
                mask = np.ones_like(frame_gray)
                # print('last trigger frame {} has boxes:'
                #       .format(last_triggered_frame_idx))
                for box in video.get_frame_detection(last_triggered_frame_idx):
                    # print('\t', box)
                    xmin, ymin, xmax, ymax, t = box[:5]
                    if t not in {COCOLabels.CAR.value, COCOLabels.BUS.value,
                                 COCOLabels.TRUCK.value}:
                        mask[ymin:ymax, xmin:xmax] = 0
                # print('previous frame {} has boxes:'.format(i-1))

                # mask off the non-viechle objects in current frame
                # for box in gt[i]:
                for box in video.get_frame_detection(i):
                    # print('\t', box)
                    xmin, ymin, xmax, ymax, t = box[:5]
                    if t not in {COCOLabels.CAR.value, COCOLabels.BUS.value,
                                 COCOLabels.TRUCK.value}:
                        mask[ymin:ymax, xmin:xmax] = 0
                # for box in dt_glimpse[i-1]:
                #     # print('\t', box)
                #     xmin, ymin, xmax, ymax, t = box[:5]
                #     if t not in [3, 8]:
                #         mask[ymin:ymax, xmin:xmax] = 0
                lastTriggeredFrameGray_masked = \
                    lastTriggeredFrameGray.copy() * mask
                frame_gray_masked = frame_gray.copy() * mask
                # compute frame difference

                frame_diff, pix_change, pix_change_obj, pix_change_bg = \
                    frame_difference(lastTriggeredFrameGray_masked,
                                     frame_gray_masked,
                                     filtered_gt[last_triggered_frame_idx],
                                     filtered_gt[i])
                pix_change_obj_list.append(pix_change_obj)
                pix_change_bg_list.append(pix_change_bg)
            else:
                # compute frame difference
                frame_diff, pix_change, pix_change_obj, pix_change_bg = \
                    frame_difference(lastTriggeredFrameGray, frame_gray,
                                     filtered_gt[last_triggered_frame_idx],
                                     filtered_gt[i])
            if frame_diff > frame_difference_thresh:
                # triggered
                # run inference to get the detection results
                dt_glimpse[i] = filtered_gt[i]
                # video.get_frame_detection(i).copy()
                frames_triggered.add(i)
                lastTriggeredFrameGray = frame_gray.copy()
                last_triggered_frame_idx = i
                debug_print('frame diff {} > {}, trigger {}, last trigger {}'
                            .format(frame_diff, frame_difference_thresh, i,
                                    last_triggered_frame_idx))

                if i in ideally_triggered_frames:
                    # print('frame diff {} > th {}, trigger frame {}, tp'
                    #       .format(frame_diff, frame_difference_thresh, i))
                    tp += 1
                    ideally_triggered_frames.remove(i)
                elif i-1 in ideally_triggered_frames:
                    tp += 1
                    ideally_triggered_frames.remove(i-1)
                elif i-2 in ideally_triggered_frames:
                    tp += 1
                    ideally_triggered_frames.remove(i-2)
                elif i-3 in ideally_triggered_frames:
                    tp += 1
                    ideally_triggered_frames.remove(i-3)
                else:
                    # print('frame diff {} > th {}, trigger extra frame {}, fp'
                    #       .format(frame_diff, frame_difference_thresh, i))
                    pass
                # if view:
                #     view = visualize(newFrameGray, pix_change, gt,
                # dt_glimpse, i,
                #                      frame_to_new_obj)
                #     # mask *= 255
                #     # view = visualize(newFrameGray, mask, gt, dt_glimpse, i,
                #     #                  frame_to_new_obj)

            else:
                # if view:
                #     view = visualize(newFrameGray, pix_change, gt,
                # dt_glimpse,
                #                      i, frame_to_new_obj)
                #     # mask *= 255
                #     # view = visualize(newFrameGray, mask, gt, dt_glimpse,
                #     #                  i, frame_to_new_obj)
                if i in ideally_triggered_frames:
                    # print('frame diff {} < th {}, miss triggering {}, fn'
                    #       .format(frame_diff, frame_difference_thresh, i))
                    # view = visualize(newFrameGray, pix_change, gt,
                    # dt_glimpse,
                    #                  i, frame_to_new_obj)
                    pass

                if self.mode == 'frame select':
                    # frame select this is used to be comparable to videostorm
                    dt_glimpse[i] = dt_glimpse[i - 1]
                elif self.mode == 'perfect tracking':
                    obj_id_in_perv_frame = [box[6] for box in dt_glimpse[i-1]]
                    # assume perfect tracking, the boxes of all objects
                    # in previously detected frame will be get the boxes
                    # in current frame from the groundtruth
                    dt_glimpse[i] = [box for box in filtered_gt[i]
                                     if box[-1] in obj_id_in_perv_frame]
                else:
                    # TODO: need to test
                    # default mode do tracking
                    # prev frame is not empty, do tracking
                    status, new_boxes = \
                        tracking_boxes(frame_gray, prev_frame_gray, frame_gray,
                                       i, dt_glimpse[i - 1],
                                       tracking_error_thresh)
                    if status:
                        # tracking is successful
                        dt_glimpse[i] = new_boxes
                    else:
                        # tracking failed and trigger a new frame
                        dt_glimpse[i] = filtered_gt[i]
                        frames_triggered.add(i)
                        lastTriggeredFrameGray = frame_gray.copy()
            prev_frame_gray = frame_gray.copy()
        # cv2.destroyAllWindows()

        f1 = eval_pipeline_accuracy(frame_start, frame_end, filtered_gt,
                                    dt_glimpse)
        fp = len(frames_triggered) - tp
        fn = len(ideally_triggered_frames)
        avg_pix_change_obj = np.mean(pix_change_obj_list)
        avg_pix_change_bg = np.mean(pix_change_bg_list)
        print('tp={}, fp={}, fn={}, nb_triggered={}, nb_ideal={}, '
              'pix_change_obj={}, pix_change_bg={}'
              .format(tp, fp, fn, len(frames_triggered), ideal_nb_triggered,
                      avg_pix_change_obj, avg_pix_change_bg))

        trigger_f1 = compute_f1(tp, fp, fn)
        return ideal_nb_triggered, f1, trigger_f1, avg_pix_change_obj, \
            avg_pix_change_bg, frames_triggered


def frame_difference(old_frame, new_frame, bboxes_last_triggered, bboxes,
                     thresh=35):
    """Compute the sum of pixel differences which are greater than thresh."""
    # thresh = 35 is used in Glimpse paper
    diff = np.absolute(new_frame.astype(int) - old_frame.astype(int))
    mask = np.greater(diff, thresh)
    pix_change_obj = 0
    obj_region = np.zeros_like(new_frame)
    for box in bboxes_last_triggered:
        xmin, ymin, xmax, ymax = box[:4]
        obj_region[ymin:ymax, xmin:xmax] = 1
        # pix_change_obj += np.sum(mask[ymin:ymax, xmin:xmax])
    for box in bboxes:
        xmin, ymin, xmax, ymax = box[:4]
        obj_region[ymin:ymax, xmin:xmax] = 1
        # pix_change_obj += np.sum(mask[ymin:ymax, xmin:xmax])
    pix_change_obj += np.sum(mask * obj_region)
    pix_change = np.sum(mask)
    pix_change_bg = pix_change - pix_change_obj

    return pix_change, (mask*255).astype(np.uint8), pix_change_obj, \
        pix_change_bg


def tracking_boxes(vis, oldFrameGray, newFrameGray, new_frame_id, old_boxes,
                   tracking_error_thresh):
    """
    Tracking the bboxes between frames via optical flow.

    Arg
        vis
        oldFrameGray
        newFrameGray
        new_frame_id
        old_boxes
        tracking_error_thresh
    Return
        tracking status(boolean) - tracking success or failure
        new bboxes tracked by optical flow
    """
    # find corners
    lk_params = dict(winSize=(15, 15), maxLevel=5,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                               10, 0.03))

    feature_params = dict(maxCorners=50, qualityLevel=0.01,
                          minDistance=7, blockSize=7)

    mask = np.zeros_like(oldFrameGray)

    old_corners = []
    for x, y, xmax, ymax, t, score, obj_id in old_boxes:
        mask[y:ymax, x:xmax] = 255
        corners = cv2.goodFeaturesToTrack(oldFrameGray[y:ymax, x:xmax],
                                          **feature_params)
        if corners is not None:
            corners[:, 0, 0] = corners[:, 0, 0] + x
            corners[:, 0, 1] = corners[:, 0, 1] + y
            old_corners.append(corners)
    if not old_corners:
        # old_corners = np.concatenate(old_corners)
        return True, []
        # cv2.imshow(str('bad'), oldFrameGray)
        # cv2.moveWindow(str('bad'), 0, 0)
        # cv2.waitKey(0)
        # cv2.destroyWindow(str('bad'))
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
    # pdb.set_trace()
    # new_corners_copy = new_corners.copy()
    new_corners = new_corners[st == 1].reshape(-1, 1, 2)
    old_corners = old_corners[st == 1].reshape(-1, 1, 2)
    # new_corners = new_corners[good]
    # old_corners = old_corners[good]

    for new_c, old_c in zip(new_corners, old_corners):
        cv2.circle(vis, (new_c[0][0], new_c[0][1]), 5, (0, 255, 255), -1)
        cv2.circle(vis, (old_c[0][0], old_c[0][1]), 5, (0, 0, 0), -1)

    new_boxes = []

    for x, y, xmax, ymax, t, score, obj_id in old_boxes:
        indices = []
        for idx, (old_c, new_c) in enumerate(zip(old_corners, new_corners)):
            # print(old_corner)
            # cv2.circle(newFrameGray, (old_corner[0][0], old_corner[0][1]),
            #            5, (255, 0, 0))
            if old_c[0][0] >= x and old_c[0][0] <= xmax and \
               old_c[0][1] >= y and old_c[0][1] <= ymax:
                indices.append(idx)
        if not indices:
            debug_print('frame {}: object {} disappear'.format(new_frame_id,
                                                               obj_id))
            continue
        indices = np.array(indices)

        old_bbox = cv2.boundingRect(old_corners[indices])
        new_bbox = cv2.boundingRect(new_corners[indices])
        cv2.rectangle(vis, (old_bbox[0], old_bbox[1]),
                      (old_bbox[0]+old_bbox[2], old_bbox[1]+old_bbox[3]),
                      (0, 0, 0), 3)
        cv2.rectangle(vis, (new_bbox[0], new_bbox[1]),
                      (new_bbox[0]+new_bbox[2], new_bbox[1]+new_bbox[3]),
                      (0, 255, 255), 3)

        # TODO: corner cleaning

        # checking tracking error threshold condition
        displacement_vectors = []
        dist_list = []
        for old_corner, new_corner in zip(old_corners[indices],
                                          new_corners[indices]):
            dist_list.append(np.linalg.norm(new_corner-old_corner))
            displacement_vectors.append(new_corner-old_corner)
        tracking_err = np.std(dist_list)
        if tracking_err > tracking_error_thresh:
            # tracking failure, this is a trigger frame
            debug_print('frame {}: '
                        'object {} std {} > tracking error thresh {}, '
                        'tracking fails'.format(new_frame_id, obj_id,
                                                np.std(dist_list),
                                                tracking_error_thresh))
            return False, []

        # update bouding box translational movement
        mean_displacement_vector = np.mean(displacement_vectors, axis=0)
        # print(mean_displacement_vector)

        # update bouding box zooming movement
        old_dists = []
        new_dists = []
        old_x_lengths = []
        old_y_lengths = []
        new_x_lengths = []
        new_y_lengths = []
        for i in range(len(old_corners[indices])):
            for j in range(i+1, len(old_corners[indices])):
                c_i = old_corners[indices][i]
                c_j = old_corners[indices][j]
                dist = np.linalg.norm(c_j-c_i)
                old_dists.append(dist)
                old_x_lengths.append(c_j[0][0]-c_i[0][0])
                old_y_lengths.append(c_j[0][1]-c_i[0][1])

                c_i = new_corners[indices][i]
                c_j = new_corners[indices][j]
                dist = np.linalg.norm(c_j-c_i)
                new_dists.append(dist)
                new_x_lengths.append(c_j[0][0]-c_i[0][0])
                new_y_lengths.append(c_j[0][1]-c_i[0][1])

        min_x_idx = np.argmin(old_corners[indices, 0, 0])
        max_x_idx = np.argmax(old_corners[indices, 0, 0])
        min_y_idx = np.argmin(old_corners[indices, 0, 1])
        max_y_idx = np.argmax(old_corners[indices, 0, 1])
        # pdb.set_trace()
        old_x_len = old_corners[indices, 0, 0][max_x_idx] - \
            old_corners[indices, 0, 0][min_x_idx]
        old_y_len = old_corners[indices, 0, 1][max_y_idx] - \
            old_corners[indices, 0, 1][min_y_idx]

        new_x_len = new_corners[indices, 0, 0][max_x_idx] - \
            new_corners[indices, 0, 0][min_x_idx]
        new_y_len = new_corners[indices, 0, 1][max_y_idx] - \
            new_corners[indices, 0, 1][min_y_idx]
        # print(old_corners[min_x_idx, 0, 0], old_corners[min_y_idx, 0, 0])

        # ratio = np.nanmean(np.array(new_dists)/np.array(old_dists))
        w_ratio = new_x_len/old_x_len
        h_ratio = new_y_len/old_y_len
        # use bbox to compute ratio
        w_ratio = new_bbox[2]/old_bbox[2]
        h_ratio = new_bbox[3]/old_bbox[3]

        if np.isnan(w_ratio):
            w_ratio = 1
        if np.isnan(h_ratio):
            h_ratio = 1

        old_center_x = (x + xmax)/2
        old_center_y = (y + ymax)/2
        new_center_x = old_center_x + mean_displacement_vector[0][0]
        new_center_y = old_center_y + mean_displacement_vector[0][1]
        # use bbox to compute displacement
        # new_center_x = old_center_x + new_bbox[0]-old_bbox[0]
        # new_center_y = old_center_y + new_bbox[1]-old_bbox[1]
        new_w = (xmax-x) * w_ratio
        new_h = (ymax-y) * h_ratio
        # new_w = ((xmax-x) + new_w)/2
        # new_h = ((ymax-y) + new_h)/2

        new_x = int(new_center_x - new_w/2)
        new_y = int(new_center_y - new_h/2)

        new_xmax = int(new_x + int(new_w))
        new_ymax = int(new_y + int(new_h))

        # new_x = int(x+mean_displacement_vector[0][0])
        # new_y = int(y+mean_displacement_vector[0][1])
        # # new_xmax = xmax
        # # new_ymax = ymax
        # new_xmax = int(new_x + (xmax - x) * ratio)
        # new_ymax = int(new_y + (ymax - y) * ratio)
        new_boxes.append([new_x, new_y, new_xmax, new_ymax, t, score, obj_id])

    return True, new_boxes


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
        tp[i], fp[i], fn[i] = eval_single_image(gt_boxes_final, dt_boxes_final,
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
    for frame_id in range(start, end+1):
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
