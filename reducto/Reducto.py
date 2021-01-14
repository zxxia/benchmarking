
import os
import pdb
from typing import List, Tuple, Union

import numpy as np

from evaluation.f1 import compute_f1, evaluate_video
from reducto.differencer import Differencer
from reducto.threshold_map import ThresholdMap
from utils.utils import read_json_file  # , write_json_file
from videos import Video


class Reducto:
    """Implementation of Reducto."""

    def __init__(self, video: Video, thresholds_file: Union[None, str] = None,
                 target_f1: float = 0.9, segment_duration: int = 1):
        """

        Args
            video: A video object.
            thresholds_file: None or the path to a json file containing all
                thresholds for all feature types.
            target_f1: target f1 score.
            segment_duration: segment length in second. 1s is used in Reductor
                paper.
        """
        self.video = video
        if thresholds_file is not None:
            self.feature_thresholds = read_json_file(thresholds_file)
        else:
            self.feature_thresholds = None
        self.segment_duration = segment_duration
        self.feat_types = ['pixel', 'area', 'edge', 'corner']
        # self.feat_types = ['pixel']  # , 'area', 'edge', 'corner']
        self.target_f1 = target_f1

        # number of segments used to build up the initial training set
        self.len_bootstrapping = 5
        self.eval_results = []
        self.diff_vectors = []

    def profile(self, frame_range: Union[None, Tuple[int, int]] = None):
        """Offline profile."""
        if frame_range is None:
            frame_range = (self.video.start_frame_index,
                           self.video.end_frame_index)
        assert self.video.start_frame_index <= frame_range[0] <= \
            self.video.end_frame_index
        assert self.video.start_frame_index <= frame_range[1] <= \
            self.video.end_frame_index
        assert frame_range[0] <= frame_range[1]

        thresholds = {}
        features = {}
        diffs = {}
        # prepare features and candidate thresholds
        for feat_type in self.feat_types:
            print('Computing candidate thresholds for {}...'.format(feat_type))
            differencer = Differencer.str2class(feat_type)()

            feats = [differencer.get_frame_feature(
                self.video.get_frame_image(i))
                for i in range(frame_range[0], frame_range[1]+1)]
            features[feat_type] = feats
            frame_diffs = [differencer.cal_frame_diff(ft1, ft0)
                           for ft0, ft1 in zip(feats[:-1], feats[1:])]
            diffs[feat_type] = frame_diffs
            sorted_frame_diffs = sorted(frame_diffs)
            thresholds[feat_type] = list(np.linspace(
                sorted_frame_diffs[0], sorted_frame_diffs[-1], num=50))

        # select the best feature for this video
        best_feat_type = None
        best_thresh = None
        smallest_perf = None
        for feat_type in self.feat_types:
            # iterate over all segments of the video
            thresh_f1s = {}  # mapping threshold to a list of f1 scores
            thresh_perfs = {}  # mapping threshold to a list of % of frame sent
            for idx in range(frame_range[0], frame_range[1] + 1,
                             self.segment_duration * self.video.frame_rate):
                seg_start = idx
                seg_end = min(seg_start + self.segment_duration *
                              self.video.frame_rate - 1, frame_range[1])
                selected_frames = select_frames(
                    diffs[feat_type][seg_start-frame_range[0]:
                                     seg_end-frame_range[0]],
                    thresholds[feat_type])

                # evaluate the f1 score of a segment for each threshold
                for thresh, frame_ids in selected_frames.items():
                    dets = {}
                    gt = {}
                    for i in range(seg_start, seg_end + 1):
                        if i - seg_start in frame_ids:
                            dets[i] = self.video.get_frame_detection(i)
                        else:
                            dets[i] = dets[i-1]
                        gt[i] = self.video.get_frame_detection(i)
                    tp, fp, fn = evaluate_video(gt, dets)
                    if thresh not in thresh_f1s:
                        thresh_f1s[thresh] = []
                    thresh_f1s[thresh].append(compute_f1(tp, fp, fn))
                    if thresh not in thresh_perfs:
                        thresh_perfs[thresh] = []
                    thresh_perfs[thresh].append(
                        len(frame_ids)/(seg_end-seg_start+1))
            for thresh in thresh_f1s:
                avg_perf = np.mean(thresh_perfs[thresh])
                if ((np.mean(thresh_f1s[thresh]) > self.target_f1) and
                        (smallest_perf is None or avg_perf < smallest_perf)):
                    best_feat_type = feat_type
                    smallest_perf = avg_perf
                    best_thresh = thresh
        print(best_feat_type, best_thresh)
        # if self.feature_thresholds is None:
        self.feature_thresholds = thresholds
        if best_feat_type is None:
            pdb.set_trace()
        assert best_feat_type is not None
        assert best_thresh is not None
        return best_feat_type, best_thresh, thresholds

    def profile_segment(self, frame_range: Union[None, Tuple[int, int]] = None):
        """Offline profile."""
        if frame_range is None:
            frame_range = (self.video.start_frame_index,
                           self.video.end_frame_index)
        assert self.video.start_frame_index <= frame_range[0] <= \
            self.video.end_frame_index
        assert self.video.start_frame_index <= frame_range[1] <= \
            self.video.end_frame_index
        assert frame_range[0] <= frame_range[1]

        thresholds = {}
        features = {}
        diffs = {}
        # prepare features and candidate thresholds
        for feat_type in self.feat_types:
            print('Computing candidate thresholds for {}...'.format(feat_type))
            differencer = Differencer.str2class(feat_type)()

            feats = [differencer.get_frame_feature(
                self.video.get_frame_image(i))
                for i in range(frame_range[0], frame_range[1]+1)]
            features[feat_type] = feats
            frame_diffs = [differencer.cal_frame_diff(ft1, ft0)
                           for ft0, ft1 in zip(feats[:-1], feats[1:])]
            diffs[feat_type] = frame_diffs
            sorted_frame_diffs = sorted(frame_diffs)
            thresholds[feat_type] = list(np.linspace(
                sorted_frame_diffs[0], sorted_frame_diffs[-1], num=50))

        # select the best feature for this video
        best_feat_type = None
        best_thresh = None
        smallest_perf = None
        feat_perf = {}
        for feat_type, threshs in thresholds.items():
            thresh_f1s = {}  # mapping threshold to a list of f1 scores
            thresh_perfs = {}  # mapping threshold to a list of % of frame sent
            feat_perf[feat_type] = {}
            for thresh in threshs:
                feat_perf[feat_type][thresh] = {'f1': 1, 'frame_rate': 1}
            # iterate over all segments of the video
                tot_tp, tot_fp, tot_fn = 0, 0, 0
                for idx in range(frame_range[0], frame_range[1] + 1,
                                 self.segment_duration * self.video.frame_rate):
                    seg_start = idx
                    seg_end = min(seg_start + self.segment_duration *
                                  self.video.frame_rate - 1, frame_range[1])
                    selected_frames = select_frames_with_thresh(
                        diffs[feat_type][seg_start-frame_range[0]:
                                         seg_end-frame_range[0]], thresh)

                    # evaluate the f1 score of a segment for each threshold
                    dets = {}
                    gt = {}
                    for i in range(seg_start, seg_end + 1):
                        if i - seg_start in selected_frames:
                            dets[i] = self.video.get_frame_detection(i)
                        else:
                            dets[i] = dets[i-1]
                        gt[i] = self.video.get_frame_detection(i)
                    tp, fp, fn = evaluate_video(gt, dets)
                    tot_tp += tp
                    tot_fp += fp
                    tot_fn += fn
                    if thresh not in thresh_f1s:
                        thresh_f1s[thresh] = []
                    thresh_f1s[thresh].append(compute_f1(tp, fp, fn))
                    if thresh not in thresh_perfs:
                        thresh_perfs[thresh] = []
                    thresh_perfs[thresh].append(
                        len(selected_frames)/(seg_end-seg_start+1))
                feat_perf[feat_type][thresh]['f1'] = compute_f1(
                    tot_tp, tot_fp, tot_fn)
                feat_perf[feat_type][thresh]['frame_rate'] = \
                    np.sum(thresh_perfs[thresh]) * self.video.frame_rate * \
                    self.segment_duration / (frame_range[1] - frame_range[0])
            for thresh in thresh_f1s:
                avg_perf = np.mean(thresh_perfs[thresh])
                if ((np.mean(thresh_f1s[thresh]) > self.target_f1) and
                        (smallest_perf is None or avg_perf < smallest_perf)):
                    best_feat_type = feat_type
                    smallest_perf = avg_perf
                    best_thresh = thresh
        # print(best_feat_type, best_thresh)
        # if self.feature_thresholds is None:
        self.feature_thresholds = thresholds
        if best_feat_type is None:
            pdb.set_trace()
        assert best_feat_type is not None
        assert best_thresh is not None
        return best_feat_type, best_thresh, feat_perf

    def evaluate(self, feat_type:str,
                 frame_range: Union[None, Tuple[int, int]] = None):
        """Online evaluate."""
        if self.feature_thresholds is None:
            raise ValueError('feature_thresholds is None.')
        if frame_range is None:
            frame_range = (self.video.start_frame_index,
                           self.video.end_frame_index)
        assert self.video.start_frame_index <= frame_range[0] <= \
            self.video.end_frame_index
        assert self.video.start_frame_index <= frame_range[1] <= \
            self.video.end_frame_index
        assert frame_range[0] <= frame_range[1]
        f1_scores = []
        perfs = []
        for idx in range(frame_range[0], frame_range[1]+1,
                         self.segment_duration * self.video.frame_rate):
            seg_start = idx
            seg_end = min(seg_start + self.segment_duration *
                          self.video.frame_rate - 1, frame_range[1])
            differencer = Differencer.str2class(feat_type)()
            feats = [differencer.get_frame_feature(
                self.video.get_frame_image(i))
                for i in range(seg_start, seg_end+1)]
            diff_vec = [differencer.cal_frame_diff(ft1, ft0)
                        for ft0, ft1 in zip(feats[:-1], feats[1:])]
            if len(self.eval_results) < self.len_bootstrapping:
                selected_frames = select_frames(
                    diff_vec, self.feature_thresholds[feat_type])
                thresh_f1s, _ = get_threshold_performance_mapping(
                    selected_frames, seg_start, seg_end, self.video)
                self.eval_results.append(thresh_f1s)
                self.diff_vectors.append(diff_vec)
                if len(self.eval_results) == self.len_bootstrapping:
                    self.thresh_map = ThresholdMap.build(
                        self.eval_results, self.diff_vectors, feat_type)
                f1_scores.append(1)
                perfs.append(1)
            else:
                thresh, distance = self.thresh_map.get_thresh(diff_vec)
                # if np.sum(distance) > dist_thresh:
                print('distance', np.sum(distance))
                if np.sum(distance) > 2.5:
                    print('update the knn')
                    selected_frames = select_frames(
                        diff_vec, self.feature_thresholds[feat_type])
                    thresh_f1s, _ = get_threshold_performance_mapping(
                        selected_frames, seg_start, seg_end, self.video)
                    self.eval_results.append(thresh_f1s)
                    self.diff_vectors.append(diff_vec)
                    f1_scores.append(1)
                    perfs.append(1)
                else:
                    selected_frames = select_frames_with_thresh(
                        diff_vec, thresh)
                    f1, perf, _ = get_threshold_performance("",
                        self.video, selected_frames, (seg_start, seg_end))
                    f1_scores.append(f1)
                    perfs.append(perf)
        return f1_scores, perfs

    def evaluate_segment(self, segment_name: str, feat_type: str,
                         feat_thresh: float,
                         frame_range: Union[None, Tuple[int, int]] = None,
                         video_save_dir: Union[None, str] = None):
        """Evaluate."""
        if self.feature_thresholds is None:
            raise ValueError('feature_thresholds is None.')
        if frame_range is None:
            frame_range = (self.video.start_frame_index,
                           self.video.end_frame_index)
        assert self.video.start_frame_index <= frame_range[0] <= \
            self.video.end_frame_index
        assert self.video.start_frame_index <= frame_range[1] <= \
            self.video.end_frame_index
        assert frame_range[0] <= frame_range[1]
        differencer = Differencer.str2class(feat_type)()
        feats = [differencer.get_frame_feature(self.video.get_frame_image(i))
                 for i in range(frame_range[0], frame_range[1]+1)]
        diff_vec = [differencer.cal_frame_diff(ft1, ft0)
                    for ft0, ft1 in zip(feats[:-1], feats[1:])]
        # thresh, distance = self.thresh_map.get_thresh(diff_vec)
        selected_frames = select_frames_with_thresh(diff_vec, feat_thresh)
        f1, relative_fps, relative_bw = get_threshold_performance(
            segment_name, self.video, selected_frames, frame_range,
            video_save_dir)
        return f1, relative_fps, relative_bw


def get_threshold_performance_mapping(selected_frames, seg_start, seg_end, video):
    thresh_f1s = {}
    thresh_perfs = {}
    for thresh, frame_ids in selected_frames.items():
        f1, perf, _ = get_threshold_performance(video, frame_ids,
                                                (seg_start, seg_end), None)
        if thresh not in thresh_f1s:
            thresh_f1s[thresh] = []
        thresh_f1s[thresh].append(f1)
        if thresh not in thresh_perfs:
            thresh_perfs[thresh] = []
        thresh_perfs[thresh].append(perf)
    return thresh_f1s, thresh_perfs


def get_threshold_performance(segment_name: str, video: Video,
                              selected_frames: List[int],
                              frame_range: Tuple[int, int],
                              video_save_dir: Union[None, str] = None):
    dets = {}
    gt = {}
    for i in range(frame_range[0], frame_range[1] + 1):
        if i - frame_range[0] in selected_frames:
            dets[i] = video.get_frame_detection(i)
        else:
            dets[i] = dets[i-1]
        gt[i] = video.get_frame_detection(i)
    tp, fp, fn = evaluate_video(gt, dets)
    relative_fps = len(selected_frames)/(frame_range[1] - frame_range[0]+1)

    # measure the bandwitdth
    if video_save_dir is not None:
        video_save_name = os.path.join(
            video_save_dir, segment_name+'_original_eval'+'.mp4')
        origin_bw = video.encode(
            video_save_name, list(range(frame_range[0], frame_range[1]+1)),
            video.frame_rate)
        video_save_name = os.path.join(
            video_save_dir, segment_name+'_eval'+'.mp4')
        bw = video.encode(video_save_name,
                          [i + frame_range[0] for i in selected_frames],
                          relative_fps * video.frame_rate)
        relative_bw = bw/origin_bw
    else:
        relative_bw = 1
    return compute_f1(tp, fp, fn), relative_fps, relative_bw


def select_frames(diff_value, thresholds):
    diff_results = {}
    for thresh in thresholds:
        diff_results[thresh] = select_frames_with_thresh(diff_value, thresh)
    return diff_results


def select_frames_with_thresh(diff_value, thresh):
    diff_integral = np.cumsum([0.0] + diff_value).tolist()
    total_frames = 1 + len(diff_value)
    selected_frames = [0]
    estimations = [1.0]
    last, current = 0, 1
    while current < total_frames:
        diff_delta = diff_integral[current] - diff_integral[last]
        if diff_delta >= thresh:
            selected_frames.append(current)
            last = current
            estimations.append(1.0)
        else:
            estimations.append((thresh - diff_delta) / thresh)
        current += 1
    return selected_frames
    # diff_results[thresh] = DiffProcessor._format_result(selected_frames,
    # total_frames, estimations)
    # return diff_results
