import copy
import csv
import os
import pdb
import sys
from collections import defaultdict

from constants import MODEL_COST
from utils.utils import interpolation
from videos import get_dataset_class, get_seg_paths
from evaluation.f1 import compute_f1, evaluate_frame

from interface import Temporal, Spatial, Model
from pipeline import Pipeline



class VideoStorm_Temporal(Temporal):
    '''use sample rate'''
    def __init__(self, temporal_sampling_list, videostorm_temporal_flag):
        '''
        :param temporal_sampling_list: flag == 1 -> sampling_list regard as sample rate; else: sample_rate = 0
        :param videostorm_temporal_flag: flag whether use videostorm_tempora pruning
        '''
        if videostorm_temporal_flag:
            # use input sample
            self.temporal_sampling_list = temporal_sampling_list
        else:
            # no sampling
            self.temporal_sampling_list = [sys.maxsize]

    def run(self, segment, decision, results):
        pass


class VideoStorm_Spacial(Spatial):
    '''get certain resolution video'''
    def __init__(self, original_resolution, spacial_resolution, videostorm_spacial_flag):
        if videostorm_spacial_flag:
            self.resolution = spacial_resolution
        else:
            self.resolution = original_resolution

    def run(self, segment, decision, results):
        pass


class VideoStorm_Model(Model):
    '''use different model'''
    def __init__(self, model_list, videostorm_spacial_flag):
        print(videostorm_spacial_flag)
        if videostorm_spacial_flag:
            self.model_list = model_list
            print("VideoStorm_Model SELECTED!!!!!!!!!!!!!!!!!!!!!!")
        else:
            self.model_list = ['faster_rcnn_resnet101', 'faster_rcnn_inception_v2', 'ssd_mobilenet_v2']
            print("VideoStorm_Model NOT SELECTED!!!!!!!!!!!!!!!!!!!!!!")

    def run(self, segment, decision, results):
        pass


class VideoStorm(Pipeline):
    def __init__(self, temporal_sampling_list, model_list, original_resolution, spacial_resolution, quantizer_list, profile_log, video_save_path, videostorm_temporal_flag, videostorm_spacial_flag, videostorm_model_flag, target_f1 = 0.9 ):
        '''
        Load the configs and initialize VideoStorm_interface pipeline.
        :param temporal_sampling_list: a list of sample rates
        :param model_list: a list of model names
        :param resolution: selected resolution
        :param target_f1: target f1 score
        :param temporal_prune: Temporal prune instance for Videostorm
        :param spacial_prune: Spacial prune instance for Videostorm
        :param model_prune: Model prune instance for Videostorm
        '''
        self.target_f1 = target_f1
        self.temporal_sampling_list = temporal_sampling_list
        self.quantizer_list = quantizer_list
        self.profile_writer = csv.writer(open(profile_log, 'w', 1))
        self.video_save_path = video_save_path
        # Add header
        self.profile_writer.writerow(['video_name', 'model', 'frame_rate', 'gpu time', 'f1'])
        # pruning flags
        self.videostorm_temporal_flag = videostorm_temporal_flag
        self.videostorm_spacial_flag = videostorm_spacial_flag
        self.videostorm_model_flag = videostorm_model_flag
        # use these to get temporal_sampling_list, model_list, resolution
        self.videostorm_temporal = VideoStorm_Temporal(temporal_sampling_list, videostorm_temporal_flag)
        self.videostorm_spacial = VideoStorm_Spacial(original_resolution, spacial_resolution, videostorm_spacial_flag)
        self.videostorm_model = VideoStorm_Model(model_list, videostorm_model_flag)


    def Source(self, clip, pruned_video_dict, original_video, frame_range):
        '''
        :param clip: video_frame
        :param pruned_video_dict: videos after pruned
        :param original_video: groundtruth
        :param frame_range: start_frame, end_frame
        :return: profiled best_frame_rate, best_model
        '''
        self.Server(clip, pruned_video_dict, original_video, frame_range)


    def Server(self, clip, pruned_video_dict, original_video, frame_range):
        '''
        :param the same as Source params
        '''
        original_gpu_time = MODEL_COST[original_video.model] * original_video.frame_rate
        min_gpu_time = original_gpu_time
        best_frame_rate = original_video.frame_rate
        best_model = original_video.model
        for model in self.videostorm_model.model_list:
            video = pruned_video_dict[model]
            f1_list = []
            for sample_rate in self.videostorm_temporal.temporal_sampling_list:
                f1_score, relative_gpu_time, _ = self.evaluate(video, original_video, sample_rate, frame_range)
                #print('{}, relative fps={:.3f}, f1={:.3f}'.format(model, 1 / sample_rate, f1_score))
                f1_list.append(f1_score)
                self.profile_writer.writerow([clip, video.model, 1 / sample_rate, relative_gpu_time, f1_score])

            frame_rate_list = [video.frame_rate / x for x in self.videostorm_temporal.temporal_sampling_list]

            if f1_list[-1] < self.target_f1:
                target_frame_rate = None
            else:
                '''
                try:
                    print(f1_list)
                    index = next(x[0] for x in enumerate(f1_list) if x[1] > self.target_f1)
                except:
                    index = 0
                '''
                index = next(x[0] for x in enumerate(f1_list) if x[1] > self.target_f1)

                if index == 0:
                    target_frame_rate = frame_rate_list[0]
                else:
                    point_a = (f1_list[index-1], frame_rate_list[index-1])
                    point_b = (f1_list[index], frame_rate_list[index])
                    target_frame_rate = interpolation(point_a, point_b, self.target_f1)

            if target_frame_rate is not None:
                gpu_time = MODEL_COST[video.model] * target_frame_rate
                if gpu_time <= min_gpu_time:
                    best_frame_rate = target_frame_rate
                    min_gpu_time = gpu_time
                    best_model = video.model

        return best_frame_rate, best_model

    def evaluate(self, clip, video, original_video, best_frame_rate, best_spacial_choice, frame_range):
        '''evaluation'''
        triggered_frames = []
        tpos = defaultdict(int)
        fpos = defaultdict(int)
        fneg = defaultdict(int)
        save_dt = []

        video_save_name = os.path.join(self.video_save_path, )
        original_bw = original_video.encode_iframe_control(video_save_name, list(range(frame_range[0], frame_range[1]+1)), original_video.frame_rate)
        sample_rate = original_video.frame_rate / best_frame_rate

        original_gpu_time = MODEL_COST[original_video.model] * original_video.frame_rate
        for img_index in range(frame_range[0], frame_range[1] + 1):
            dt_box_final = []
            current_full_model_dt = video.get_frame_detection(img_index)
            current_gt = original_video.get_frame_detection(img_index)
            # based on sample rate, decide whether this frame is sampled
            if img_index % sample_rate >= 1:
                # this frame is not sampled, so reuse the last saved detection result
                dt_box_final = copy.deepcopy(save_dt)
            else:
                # this frame is sampled, so use the full model result
                dt_box_final = copy.deepcopy(current_full_model_dt)
                save_dt = copy.deepcopy(dt_box_final)
                triggered_frames.append(img_index)
            # each frame has different types to calculate
            tpos[img_index], fpos[img_index], fneg[img_index] = evaluate_frame(current_gt, dt_box_final)

        tp_total = sum(tpos.values())
        fp_total = sum(fpos.values())
        fn_total = sum(fneg.values())

        f1_score = compute_f1(tp_total, fp_total, fn_total)
        gpu_time = MODEL_COST[video.model] * video.frame_rate / sample_rate

        video_save_name = os.path.join(self.video_save_path, clip+'_eval'+'.mp4')
        bandwidth = video.encode_iframe_control(video_save_name, triggered_frames, best_frame_rate)
        return f1_score, gpu_time / original_gpu_time, bandwidth / original_bw