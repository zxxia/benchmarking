import copy
import csv
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
        if videostorm_spacial_flag:
            self.model_list = model_list
        else:
            self.model_list = ['FasterRCNN', 'inception', 'mobilenet']

    def run(self, segment, decision, results):
        pass


class VideoStorm(Pipeline):
    def __init__(self, temporal_sampling_list, model_list, original_resolution, spacial_resolution, profile_log, target_f1 = 0.9, videostorm_temporal_flag = 0, videostorm_spacial_flag = 0, videostorm_model_flag = 0):
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
        self.profile_writer = csv.writer(open(profile_log, 'w', 1))
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
        :param frame_range: video frame range
        :return: best_frame_rate, best_model
        '''
        self.Server(clip, pruned_video_dict, original_video, frame_range)


    def Server(self, clip, pruned_video_dict, original_video, frame_range):
        '''
        :param the same as Source params
        '''





    def evaluate(self, video, original_video, sample_rate, frame_range):