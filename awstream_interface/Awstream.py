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

    def Source(self):

    def Server(self):

    def evaluate(self):