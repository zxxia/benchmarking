"""Noscope Definition."""
import copy
import csv
import os
from collections import defaultdict

import cv2
import numpy as np

from constants import MODEL_COST
from evaluation.f1 import compute_f1, evaluate_frame


class NoScope():
    """NoScope Pipeline."""

    def __init__(self, confidence_score_list, mse_thresh_list, profile_log,
                 target_f1=0.9):
        """Load the configs."""
        self.target_f1 = target_f1
        self.confidence_score_list = confidence_score_list
        self.mse_thresh_list = mse_thresh_list  # used for frame difference
        self.profile_writer = csv.writer(open(profile_log, 'w', 1))
        self.profile_writer.writerow(
            ["video_name", "mse_thresh", "confidence_score_thresh", "f1",
                "triggered_frame", "tp", "fp", "fn"])

    def profile(self, video_name, original_video, small_model_video,
                frame_range, profile_video_savepath, cost='gpu'):
        """Profile to get the confidence score threhold for target f1.

        Return
            a list of config that satisfys the requirements.
        """
        original_bw = original_video.encode(
            os.path.join(profile_video_savepath, video_name+'.mp4'),
            list(range(frame_range[0], frame_range[1])),
            original_video.frame_rate, save_video=True)
        f1_dict = {}
        bw_dict = {}
        gpu_dict = {}
        if self.mse_thresh_list == []:
            # largest possible MSE, i.e., no frame difference detector
            self.mse_thresh_list = [0]

        frame_difference = frame_diffence_compute(frame_range, original_video)
        for mse_thresh in self.mse_thresh_list:
            selected_frames = frame_difference_detector(
                frame_range, original_video.frame_rate, frame_difference,
                mse_thresh)
            for thresh in self.confidence_score_list:
                # choose confidence score
                tp_total, fp_total, fn_total, trigger_frame_list = eval_images(
                    frame_range, selected_frames, original_video,
                    small_model_video, thresh)
                f1_score = compute_f1(tp_total, fp_total, fn_total)

                self.profile_writer.writerow(
                    [video_name, mse_thresh, thresh, f1_score,
                     ' '.join([str(_x) for _x in trigger_frame_list]),
                     tp_total, fp_total, fn_total])

                f1_dict[(mse_thresh, thresh)] = f1_score
                # compute corresponding bw based on triggered frames
                bw = compute_bw(original_video, trigger_frame_list)
                abs_gpu = (len(selected_frames) * MODEL_COST['mobilenet'] +
                           len(trigger_frame_list) * MODEL_COST['FasterRCNN'])
                gpu_dict[(mse_thresh, thresh)] = (
                    abs_gpu / ((frame_range[1] + 1 - frame_range[0])) *
                    MODEL_COST['FasterRCNN'])
                bw_dict[(mse_thresh, thresh)] = bw
                print('profile on {} {}, mse thresh={}, confidence score '
                      'thresh={}, trigged frame cn={}, f1={}, bw={}'
                      .format(video_name, frame_range, mse_thresh, thresh,
                              len(trigger_frame_list), f1_score, bw))

        if cost == 'bw':
            # cost is bandwidth
            cost_dict = bw_dict
        else:
            # cost is gpu
            cost_dict = gpu_dict

        best_mse_thresh, best_thresh = find_best_thresh(
            f1_dict, cost_dict, self.target_f1)
        if (best_mse_thresh, best_thresh) in cost_dict:
            best_bw = bw_dict[(best_mse_thresh, best_thresh)]
            best_f1 = f1_dict[(best_mse_thresh, best_thresh)]
            best_gpu = gpu_dict[(best_mse_thresh, best_thresh)]
        else:
            best_bw = 1
            best_f1 = 1
            best_gpu = ((MODEL_COST['mobilenet'] + MODEL_COST['FasterRCNN']) /
                        float(MODEL_COST['FasterRCNN']))
        print("best mse thresh {}, best score thresh {}, bw {}".format(
            best_mse_thresh, best_thresh, best_bw))
        best_relative_bw = best_bw / original_bw
        return best_mse_thresh, best_thresh, best_f1, best_relative_bw, best_gpu

    def evaluate(self, video_name, original_video, small_model_video,
                 best_mse_thresh, best_thresh, frame_range):
        """Evaluate the performance of best config."""
        original_bw = original_video.encode(video_name,
                                            list(range(frame_range[0],
                                                       frame_range[1]+1)),
                                            original_video.frame_rate)

        frame_difference = frame_diffence_compute(frame_range, original_video)
        selected_frames = frame_difference_detector(
            frame_range, original_video.frame_rate, frame_difference,
            best_mse_thresh)
        tp_total, fp_total, fn_total, trigger_frame_list = eval_images(
            frame_range, selected_frames, original_video, small_model_video,
            best_thresh)
        f1_score = compute_f1(tp_total, fp_total, fn_total)
        bw = compute_bw(original_video, trigger_frame_list)

        gpu = len(selected_frames) * MODEL_COST['mobilenet'] + \
            len(trigger_frame_list) * MODEL_COST['FasterRCNN']
        relative_gpu = gpu / \
            ((frame_range[1] + 1 - frame_range[0]) * MODEL_COST['FasterRCNN'])
        return f1_score, bw / original_bw, relative_gpu, selected_frames, trigger_frame_list


def eval_images(image_range, selected_frames, original_video, video,
                thresh=0.8):
    """Evaluate the tp, fp, fn of a range of images."""
    # sample_rate = round(original_config.fps/target_config.fps)
    tpos = defaultdict(int)
    fpos = defaultdict(int)
    fneg = defaultdict(int)
    gtruth = original_video.get_video_detection()
    dets = video.get_video_detection()
    trigger_frame_list = []
    pipeline_dets = {}
    for idx in selected_frames:
        current_dt = copy.deepcopy(dets[idx])
        current_gt = copy.deepcopy(gtruth[idx])
        score_list = [x[5] for x in current_dt]
        if score_list != []:
            if np.min(score_list) < thresh:
                # trigger full model
                pipeline_dets[idx] = copy.deepcopy(current_gt)
                trigger_frame_list.append(idx)
            else:
                pipeline_dets[idx] = copy.deepcopy(current_dt)
        else:
            # if no object does not trigger full model
            pipeline_dets[idx] = copy.deepcopy(current_dt)
            # if no object will trigger full model
            # pipeline_dets[idx] = copy.deepcopy(current_gt)
            # trigger_frame_list.append(idx)

    for idx in range(image_range[0], image_range[1] + 1):
        if idx not in dets or idx not in gtruth:
            continue
        # find the previous selected frame
        previous_selected_frame_idx = max(
            [x for x in selected_frames if x <= idx])

        current_dt = copy.deepcopy(pipeline_dets[previous_selected_frame_idx])
        current_gt = copy.deepcopy(gtruth[idx])
        tpos[idx], fpos[idx], fneg[idx] = \
            evaluate_frame(current_gt, current_dt)

        # print(idx, tpos[idx], fpos[idx], fneg[idx])
    return sum(tpos.values()), sum(fpos.values()), sum(fneg.values()), trigger_frame_list


def compute_bw(original_video, trigger_frame_list):
    bw = 0
    for triggered_frame_idx in trigger_frame_list:
        bw += original_video.get_frame_filesize(triggered_frame_idx)
    return bw


def find_best_thresh(f1_dict, cost_dict, target_f1):
    if max(f1_dict.values()) < target_f1:
        best_config = (0, 1)
    else:
        best_config = None
        min_cost = max(cost_dict.values()) + 1
        for key in sorted(f1_dict):
            if f1_dict[key] >= target_f1 and cost_dict[key] < min_cost:
                best_config = key
                min_cost = cost_dict[key]

        # index = next(x[0] for x in enumerate(f1_list)
        #                     if x[1] >= target_f1)
    return best_config[0], best_config[1]


def frame_diffence_compute(frame_range, video, t_skip=0, t_diff=0.1):
    start_frame = frame_range[0]
    end_frame = frame_range[1]
    frame_diff = {}
    if t_skip != 0:
        frame_skip = round(video.frame_rate * t_skip)
    else:
        frame_skip = 1

    previous_frame_interval = round(video.frame_rate * t_diff)
    for frame_idx in range(start_frame, end_frame, frame_skip):
        previous_frame_idx = frame_idx - previous_frame_interval
        if previous_frame_idx < start_frame:
            continue
        else:
            _mse = compute_diff(video.get_frame_image_name(previous_frame_idx),
                                video.get_frame_image_name(frame_idx))
            frame_diff[frame_idx] = _mse
    return frame_diff


def frame_difference_detector(frame_range, frame_rate, frame_difference,
                              mse_thresh, t_skip=0, t_diff=0.1):
    selected_frame = []
    start_frame = frame_range[0]
    end_frame = frame_range[1]
    if t_skip != 0:
        frame_skip = round(frame_rate * t_skip)
    else:
        frame_skip = 1

    previous_frame_interval = round(frame_rate * t_diff)
    for frame_idx in range(start_frame, end_frame, frame_skip):
        previous_frame_idx = frame_idx - previous_frame_interval
        if previous_frame_idx < start_frame:
            selected_frame.append(frame_idx)
        else:
            _mse = frame_difference[frame_idx]
            if _mse > mse_thresh:
                selected_frame.append(frame_idx)
    return selected_frame


def compute_diff(previous_img, current_img):
    """Compute difference between images.

    Load a background image (or previous image), then compute the difference
    (MSE) between current image and background image. If
    difference < threshold, this frame should be
    skipped.
    """
    pre_img = cv2.imread(previous_img)
    pre_img = cv2.cvtColor(pre_img, cv2.COLOR_BGR2GRAY)
    img = cv2.imread(current_img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    MSE = mse(pre_img, img)
    return MSE


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
