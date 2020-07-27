"""Offline version of AWStream Interface implementation."""
import copy
import csv
import os
import pdb
import sys
from collections import defaultdict

from evaluation.f1 import compute_f1, evaluate_frame
from utils.utils import interpolation

from interface import Temporal, Spatial, Model
from pipeline import Pipeline



class Awstream_Temporal(Temporal):
    '''use sample rate'''
    def __init__(self, temporal_sampling_list, awstream_temporal_flag):
        '''
        :param temporal_sampling_list: flag == 1 -> sampling_list regard as sample rate; else: sample_rate = 0
        :param awstream_temporal_flag: flag whether use awstream_tempora pruning
        '''
        if awstream_temporal_flag:
            # use input sample
            self.temporal_sampling_list = temporal_sampling_list
        else:
            # no sampling
            self.temporal_sampling_list = [sys.maxsize]

    def run(self, segment, decision, results):
        pass


class Awstream_Spacial(Spatial):
    '''get certain resolution video'''
    def __init__(self, original_resolution, spacial_resolution, awstream_spacial_flag):
        if awstream_spacial_flag:
            self.resolution = spacial_resolution
        else:
            self.resolution = original_resolution

    def run(self, segment, decision, results):
        pass


class Awstream(Pipeline):
    def __init__(self, temporal_sampling_list, resolution_list, spacial_resolution_list, quantizer_list, profile_log, video_save_path, awstream_temporal_flag, awstream_spacial_flag, target_f1=0.9):
        '''Load the configs'''
        self.target_f1 = target_f1
        self.temporal_sampling_list = temporal_sampling_list
        self.resolution_list = resolution_list
        self.quantizer_list = quantizer_list
        self.profile_writer = csv.writer(open(profile_log, 'w', 1))
        self.profile_writer.writerow(['video_name', 'resolution', 'frame_rate', 'f1', 'tp', 'fp', 'fn'])
        self.video_save_path = video_save_path
        # pruned flags
        self.awstream_temporal_flag = awstream_temporal_flag
        self.awstream_spacial_flag = awstream_spacial_flag
        # use these to get temporal_sampling_list, resolution
        self.awstream_temporal = Awstream_Temporal(temporal_sampling_list, awstream_temporal_flag)
        self.awstream_spacial = Awstream_Spacial(resolution_list, spacial_resolution_list, awstream_spacial_flag)

    def Source(self, clip, video_dict, original_video, frame_range):
        """Profile the combinations of fps and resolution.

        Return a list of config that satisfys the requirements.
        """
        self.Server(clip, video_dict, original_video, frame_range)


    def Server(self, clip, video_dict, original_video, frame_range):
        """Profile the combinations of fps and resolution.

        Return a list of config that satisfys the requirements.
        """
        # videos encoded in profiler are not saved
        video_save_name = os.path.join(self.video_save_path, clip + '_original_profile' + '.mp4')
        original_bw = original_video.encode_iframe_control(video_save_name, list(range(frame_range[0], frame_range[1])), original_video.frame_rate, save_video=False)
        best_resol = original_video.resolution
        best_fps = original_video.frame_rate
        min_bw = original_bw
        for resolution in self.awstream_spacial.resolution:
            # choose resolution
            f1_list = []
            if resolution not in video_dict:
                continue
            video = video_dict[resolution]
            print('profile [{}, {}], resolution={}, original resolution={}'.format(frame_range[0], frame_range[1], video.resolution, original_video.resolution))

            for sample_rate in self.awstream_temporal.temporal_sampling_list:
                # choose frame rate
                tp_total, fp_total, fn_total = eval_images(frame_range, original_video, video, sample_rate)
                f1_score = compute_f1(tp_total, fp_total, fn_total)

                self.profile_writer.writerow([clip, resolution, 1 / sample_rate, f1_score, tp_total, fp_total, fn_total])
                print('profile on {} {}, resolution={}, sample_rate={}, f1={}'.format(clip, frame_range, resolution, sample_rate, f1_score))
                f1_list.append(f1_score)

            fps_list = [original_video.frame_rate / x for x in self.awstream_temporal.temporal_sampling_list]
            # use interpolation to find the frame rate closest to target frame
            # rate
            target_fps = find_target_fps(f1_list, fps_list, self.target_f1)
            print('Resolution={} and target fps={}'.format(resolution, target_fps))

            if target_fps is not None:
                sample_rate = original_video.frame_rate / target_fps
                target_frame_indices = []
                for img_index in range(frame_range[0], frame_range[1]+1):
                    # based on sample rate,decide whether this frame is sampled
                    if img_index % sample_rate >= 1:
                        continue
                    target_frame_indices.append(img_index)

                video_save_name = os.path.join(self.video_save_path, clip+'_profile'+'.mp4')
                bandwidth = video.encode_iframe_control(video_save_name, target_frame_indices, target_fps, save_video=False)
                print(min_bw, bandwidth)
                if bandwidth <= min_bw:
                    min_bw = bandwidth
                    best_resol = video.resolution
                    best_fps = video.frame_rate / sample_rate

        best_relative_bw = min_bw / original_bw
        return best_resol, best_fps, best_relative_bw

    def evaluate(self, clip, original_video, video, best_frame_rate, frame_range):
        """Evaluate the performance of best config."""
        video_save_name = os.path.join(self.video_save_path, clip+'_original_eval'+'.mp4')
        original_bw = original_video.encode_iframe_control(video_save_name, list(range(frame_range[0], frame_range[1]+1)), original_video.frame_rate)
        sample_rate = original_video.frame_rate / best_frame_rate
        target_frame_indices = []
        for img_index in range(frame_range[0], frame_range[1]+1):
            # based on sample rate,decide whether this frame is sampled
            if img_index % sample_rate >= 1:
                continue
            target_frame_indices.append(img_index)
        video_save_name = os.path.join(self.video_save_path, clip+'_eval'+'.mp4')
        bandwidth = video.encode_iframe_control(video_save_name, target_frame_indices, best_frame_rate)
        tp_total, fp_total, fn_total = eval_images(frame_range, original_video, video, sample_rate)

        return compute_f1(tp_total, fp_total, fn_total), bandwidth / original_bw


def find_target_fps(f1_list, fps_list, target_f1):
    """Use interpolation to find the ideal fps at target f1."""
    if target_f1 - 0.02 <= f1_list[-1] < target_f1:
        target_fps = fps_list[-1]
    elif f1_list[-1] < target_f1:
        target_fps = None
    else:
        try:
            index = next(x[0] for x in enumerate(f1_list) if x[1] >= target_f1)
        except StopIteration:
            pdb.set_trace()

        if index == 0:
            target_fps = fps_list[0]
        else:
            point_a = (f1_list[index-1], fps_list[index-1])
            point_b = (f1_list[index], fps_list[index])
            target_fps = interpolation(point_a, point_b, target_f1)

    return target_fps


def eval_images(image_range, original_video, video, sample_rate):
    """Evaluate the tp, fp, fn of a range of images."""
    # sample_rate = round(original_config.fps/target_config.fps)
    tpos = defaultdict(int)
    fpos = defaultdict(int)
    fneg = defaultdict(int)
    save_dt = []
    gtruth = original_video.get_video_detection()
    dets = video.get_video_detection()

    for idx in range(image_range[0], image_range[1]+1):
        if idx not in dets or idx not in gtruth:
            continue
        current_dt = copy.deepcopy(dets[idx])
        current_gt = copy.deepcopy(gtruth[idx])
        # based on sample rate, decide whether this frame is sampled
        if idx % sample_rate >= 1:
            # so reuse the last saved if no sampled
            dt_boxes_final = copy.deepcopy(save_dt)
        else:
            # sampled, so use the full model result
            dt_boxes_final = copy.deepcopy(current_dt)
            save_dt = copy.deepcopy(dt_boxes_final)

        # scale gt boxes
        current_gt = scale_boxes(current_gt, original_video.resolution, video.resolution)
        tpos[idx], fpos[idx], fneg[idx] = evaluate_frame(current_gt, dt_boxes_final)

    return sum(tpos.values()), sum(fpos.values()), sum(fneg.values())

def scale_boxes(boxes, in_resol, out_resol):
    """Scale a list of boxes."""
    return [scale(box, in_resol, out_resol) for box in boxes]

def scale(box, in_resol, out_resol):
    """Scale the box at input resolution to output resolution.

    Args
        box: [x, y, w, h]
        in_resl: (width, height)
        out_resl: (width, height)

    """
    assert len(box) >= 4
    ret_box = box.copy()
    x_scale = out_resol[0] / in_resol[0]
    y_scale = out_resol[1] / out_resol[1]
    ret_box[0] = int(box[0] * x_scale)
    ret_box[1] = int(box[1] * y_scale)
    ret_box[2] = int(box[2] * x_scale)
    ret_box[3] = int(box[3] * y_scale)
    return ret_box


