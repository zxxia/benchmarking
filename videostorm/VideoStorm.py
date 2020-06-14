"""VideoStrom Offline Implementation."""
import copy
import csv
from collections import defaultdict

from constants import MODEL_COST
from utils.utils import interpolation
from evaluation.f1 import compute_f1, evaluate_frame


class VideoStorm():
    """VideoStorm Pipeline."""

    def __init__(self, temporal_sampling_list, model_list, profile_log,
                 target_f1=0.9):
        """Load the configs and initialize VideoStorm pipeline.

        Args
            temporal_sampling_list(list): a list of sample rates
            model_list(list): a list of model names
            profile_log(string): a csv file that profiling information will be
                                 recorded
            target_f1(float): target f1 score

        """
        self.target_f1 = target_f1
        self.temporal_sampling_list = temporal_sampling_list
        self.profile_writer = csv.writer(open(profile_log, 'w', 1))
        # Add header
        self.profile_writer.writerow(
            ["video_name", "model", "frame_rate", "gpu time", "f1"])
        self.model_list = model_list

    def profile(self, video_name, video_dict, original_video, frame_range):
        """Profile a list of frame rate.

        Args
            video_name(string): video name
            video_dict(dict): model-video object pairs
            original_video(video object): serve as groudtruth
            framge_rage(list/tuple): start_frame, end_frame
        Return
            best_frame_rate(float): profiled best frame rate
            best_model(float): profiled best frame rate

        """
        original_gpu_time = MODEL_COST[original_video.model] * \
            original_video.frame_rate
        min_gpu_time = original_gpu_time
        best_frame_rate = original_video.frame_rate
        best_model = original_video.model
        for model in self.model_list:
            video = video_dict[model]
            f1_list = []
            for sample_rate in self.temporal_sampling_list:
                f1_score, relative_gpu_time, _ = self.evaluate(
                    video, original_video, sample_rate, frame_range)
                print('{}, relative fps={:.3f}, f1={:.3f}'.format(
                    model, 1/sample_rate, f1_score))
                f1_list.append(f1_score)
                self.profile_writer.writerow(
                    [video_name, video.model, 1 / sample_rate,
                     relative_gpu_time, f1_score])

            frame_rate_list = [video.frame_rate /
                               x for x in self.temporal_sampling_list]

            if f1_list[-1] < self.target_f1:
                target_frame_rate = None
                # target_frame_rate = video.frame_rate
            else:
                index = next(x[0] for x in enumerate(f1_list)
                             if x[1] > self.target_f1)
                if index == 0:
                    target_frame_rate = frame_rate_list[0]
                else:
                    point_a = (f1_list[index-1], frame_rate_list[index-1])
                    point_b = (f1_list[index], frame_rate_list[index])
                    target_frame_rate = interpolation(
                        point_a, point_b, self.target_f1)

            # select best profile
            if target_frame_rate is not None:
                gpu_time = MODEL_COST[video.model]*target_frame_rate
                if gpu_time <= min_gpu_time:
                    best_frame_rate = target_frame_rate
                    min_gpu_time = gpu_time
                    best_model = video.model

        return best_frame_rate, best_model

    def evaluate(self, video, original_video, sample_rate, frame_range):
        """Evaluation."""
        triggered_frames = []
        tpos = defaultdict(int)
        fpos = defaultdict(int)
        fneg = defaultdict(int)
        save_dt = []

        original_gpu_time = MODEL_COST[original_video.model] * \
            original_video.frame_rate
        for img_index in range(frame_range[0],  frame_range[1] + 1):
            dt_boxes_final = []
            current_full_model_dt = video.get_frame_detection(img_index)
            current_gt = original_video.get_frame_detection(img_index)
            # based on sample rate, decide whether this frame is sampled
            if img_index % sample_rate >= 1:
                # this frame is not sampled, so reuse the last saved
                # detection result
                dt_boxes_final = copy.deepcopy(save_dt)
            else:
                # this frame is sampled, so use the full model result
                dt_boxes_final = copy.deepcopy(current_full_model_dt)
                save_dt = copy.deepcopy(dt_boxes_final)
                triggered_frames.append(img_index)

            tpos[img_index], fpos[img_index], fneg[img_index] = \
                evaluate_frame(current_gt, dt_boxes_final)

        tp_total = sum(tpos.values())
        fp_total = sum(fpos.values())
        fn_total = sum(fneg.values())

        f1_score = compute_f1(tp_total, fp_total, fn_total)
        gpu_time = MODEL_COST[video.model] * video.frame_rate / sample_rate
        return f1_score, gpu_time/original_gpu_time, triggered_frames
