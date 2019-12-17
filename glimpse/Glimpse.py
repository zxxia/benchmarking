"""Glimpse Definition."""
import csv
import pdb
from collections import defaultdict
import numpy as np
import cv2
from benchmarking.utils.model_utils import eval_single_image
from benchmarking.utils.utils import compute_f1, interpolation

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
        # TODO: obtain filtered and kept bboxes from filter_video_detections
        frames_logs = []
        # frame id, frame diff triggered, frame diff, tracking triggered, [boxes], [tracking errors]
        obj_to_frame_range, frame_to_new_obj = object_appearance(
            frame_start, frame_end, video.get_video_detection())
        ideally_triggered_frames = set()
        for obj_id in obj_to_frame_range:
            frame_range = obj_to_frame_range[obj_id]
            ideally_triggered_frames.add(frame_range[0])
        ideal_nb_triggered = len(ideally_triggered_frames)

        dt_glimpse = defaultdict(list)
        frames_triggered = set()
        # start from the first frame
        # The first frame has to be sent to server.

        frame_gray = cv2.cvtColor(video.get_frame_image(frame_start),
                                  cv2.COLOR_BGR2GRAY)
        assert frame_gray is not None, 'cannot read image {}'.format(
            frame_start)
        lastTriggeredFrameGray = frame_gray.copy()
        prev_frame_gray = frame_gray.copy()
        last_triggered_frame_idx = frame_start

        # get detection from server
        dt_glimpse[frame_start] = video.get_frame_detection(frame_start)
        pix_change = np.zeros_like(frame_gray)
        frames_triggered.add(frame_start)
        tp = 0

        pix_change_obj_list = list()
        pix_change_bg_list = list()

        # run the pipeline for the rest of the frames
        for i in range(frame_start + 1, frame_end + 1):
            frame_bgr = video.get_frame_image(i)
            frame_gray = cv2.cvtColor(frame_bgr,
                                      cv2.COLOR_BGR2GRAY)
            assert frame_gray is not None, 'cannot read {}'.format(i)

            pix_change = np.zeros_like(frame_gray)
            # mask out bboxes
            if mask_flag:
                mask = np.ones_like(frame_gray)
                for box in video.get_dropped_frame_detection(
                        last_triggered_frame_idx):
                    xmin, ymin, xmax, ymax, t = box[:5]
                    mask[ymin:ymax, xmin:xmax] = 0

                # mask off the non-viechle objects in current frame
                for box in video.get_dropped_frame_detection(i):
                    xmin, ymin, xmax, ymax, t = box[:5]
                    mask[ymin:ymax, xmin:xmax] = 0
                lastTriggeredFrameGray_masked = \
                    lastTriggeredFrameGray.copy() * mask
                frame_gray_masked = frame_gray.copy() * mask
                # compute frame difference

                frame_diff, pix_change, pix_change_obj, pix_change_bg = \
                    frame_difference(lastTriggeredFrameGray_masked,
                                     frame_gray_masked,
                                     video.get_frame_detection(
                                         last_triggered_frame_idx),
                                     video.get_frame_detection(i))
                pix_change_obj_list.append(pix_change_obj)
                pix_change_bg_list.append(pix_change_bg)
            else:
                # compute frame difference
                frame_diff, pix_change, pix_change_obj, pix_change_bg = \
                    frame_difference(lastTriggeredFrameGray, frame_gray,
                                     video.get_frame_detection(
                                         last_triggered_frame_idx),
                                     video.get_frame_detection(i))
            if frame_diff > frame_difference_thresh:
                # triggered
                # run inference to get the detection results
                dt_glimpse[i] = video.get_frame_detection(i)
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
                    dt_glimpse[i] = [box for box in
                                     video.get_frame_detection(i)
                                     if box[-1] in obj_id_in_perv_frame]
                else:
                    # default mode do tracking
                    # prev frame is not empty, do tracking
                    status, new_boxes = \
                        tracking_boxes(frame_bgr, prev_frame_gray, frame_gray,
                                       i, dt_glimpse[i - 1],
                                       tracking_error_thresh)
                    if status:
                        # tracking is successful
                        dt_glimpse[i] = new_boxes
                    else:
                        # tracking failed and trigger a new frame
                        dt_glimpse[i] = video.get_frame_detection(i)
                        frames_triggered.add(i)
                        lastTriggeredFrameGray = frame_gray.copy()
            prev_frame_gray = frame_gray.copy()
        # cv2.destroyAllWindows()

        f1 = eval_pipeline_accuracy(frame_start, frame_end,
                                    video.get_video_detection(), dt_glimpse)
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
    lk_params = dict(winSize=(15, 15), maxLevel=5,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                               10, 0.03))

    # define good feature compuration parameters
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
        # cannot find available corners and treat as objects disappears
        return True, []
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
        print('tracking error:', tracking_err)
        if tracking_err > tracking_error_thresh:
            # tracking failure, this is a trigger frame
            debug_print('frame {}: '
                        'object {} std {} > tracking error thresh {}, '
                        'tracking fails'.format(new_frame_id, obj_id,
                                                np.std(dist_list),
                                                tracking_error_thresh))
            return False, []

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
        cv2.rectangle(vis, (x, y), (xmax, ymax), black, 2)
        cv2.rectangle(vis, (new_x, new_y), (new_xmax, new_ymax), yellow, 2)

    # img_title = 'frame {}'.format(new_frame_id)
    # cv2.imshow(img_title, vis)
    # cv2.moveWindow(img_title, 0, 0)
    # if cv2.waitKey(0) & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()
    # else:
    #     cv2.destroyWindow(img_title)
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


def compute_target_frame_rate(frame_rate_list, f1_list, target_f1=0.9):
    '''Compute target frame rate when target f1 is achieved.'''
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


def load_glimpse_profile(filename):
    """Load glimplse profile file."""
    videos = []
    para1_dict = defaultdict(list)
    perf_dict = defaultdict(list)
    acc_dict = defaultdict(list)
    with open(filename, 'r') as f_gl:
        f_gl.readline()  # remove headers
        for line in f_gl:
            cols = line.strip().split(',')
            video = cols[0]
            if video not in videos:
                videos.append(video)
            perf_dict[video].append(float(cols[4]))
            acc_dict[video].append(float(cols[3]))
            para1_dict[video].append(float(cols[1]))

    return videos, perf_dict, acc_dict, para1_dict


def load_glimpse_results(filename):
    """Load glimplse result file."""
    video_clips = []
    f1_score = []
    perf = []
    ideal_perf = []
    trigger_f1 = []
    with open(filename, 'r') as f_glimpse:
        f_glimpse.readline()
        for line in f_glimpse:
            # print(line)
            cols = line.strip().split(',')
            video_clips.append(cols[0])
            f1_score.append(float(cols[3]))
            perf.append(float(cols[4]))
            if len(cols) == 6:
                ideal_perf.append(float(cols[5]))
            if len(cols) == 7:
                ideal_perf.append(float(cols[5]))
                trigger_f1.append(float(cols[6]))
    # if len(cols) == 6:
    #     return video_clips, perf, f1_score, ideal_perf
    # if len(cols) == 7:
    #     return video_clips, perf, f1_score, ideal_perf, trigger_f1
    return video_clips, perf, f1_score
