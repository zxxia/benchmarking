"""VideoStrom Definition."""
import copy
from collections import defaultdict
from utils.model_utils import eval_single_image
from utils.utils import interpolation, compute_f1


class VideoStorm():
    """VideoStorm Pipeline."""

    def __init__(self, temporal_sampling_list, profile_log,
                 target_f1=0.9):
        """Load the configs."""
        self.target_f1 = target_f1
        self.temporal_sampling_list = temporal_sampling_list
        self.f_profile = open(profile_log, 'w', 1)
        self.f_profile.write("video_name,frame_rate,f1\n")

    def profile(self, video_name, gt, start_frame, end_frame, frame_rate):
        """Profile a list of frame rate."""
        f1_list = []
        for sample_rate in self.temporal_sampling_list:
            tpos = defaultdict(int)
            fpos = defaultdict(int)
            fneg = defaultdict(int)
            save_dt = []

            for img_index in range(start_frame, end_frame+1):
                dt_boxes_final = []
                current_full_model_dt = gt[img_index]
                current_gt = gt[img_index]
                # based on sample rate, decide whether this frame is sampled
                if img_index % sample_rate >= 1:
                    # this frame is not sampled, so reuse the last saved
                    # detection result
                    dt_boxes_final = copy.deepcopy(save_dt)
                else:
                    # this frame is sampled, so use the full model result
                    dt_boxes_final = copy.deepcopy(current_full_model_dt)
                    save_dt = copy.deepcopy(dt_boxes_final)

                tpos[img_index], fpos[img_index], fneg[img_index] = \
                    eval_single_image(current_gt, dt_boxes_final)

            tp_total = sum(tpos.values())
            fp_total = sum(fpos.values())
            fn_total = sum(fneg.values())

            f1_score = compute_f1(tp_total, fp_total, fn_total)
            print('relative fps={}, f1={}'.format(1/sample_rate, f1_score))
            f1_list.append(f1_score)
            self.f_profile.write(','.join([video_name, str(1/sample_rate),
                                           str(f1_score)])+'\n')

        frame_rate_list = [frame_rate/x for x in self.temporal_sampling_list]

        if f1_list[-1] < self.target_f1:
            target_frame_rate = None
            target_frame_rate = frame_rate
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
        smallest_gpu_time = 100*frame_rate
        gpu_time = 100*target_frame_rate

        if gpu_time <= smallest_gpu_time:
            best_frame_rate = target_frame_rate

        return best_frame_rate

    def evaluate(self, gt, best_sample_rate, start_frame, end_frame):
        """Evaluation."""
        triggered_frames = []
        tpos = defaultdict(int)
        fpos = defaultdict(int)
        fneg = defaultdict(int)
        save_dt = []

        for img_index in range(start_frame, end_frame + 1):
            dt_boxes_final = []
            current_full_model_dt = gt[img_index]
            current_gt = gt[img_index]
            # based on sample rate, decide whether this frame is sampled
            if img_index % best_sample_rate >= 1:
                # this frame is not sampled, so reuse the last saved
                # detection result
                dt_boxes_final = copy.deepcopy(save_dt)
            else:
                # this frame is sampled, so use the full model result
                dt_boxes_final = copy.deepcopy(current_full_model_dt)
                save_dt = copy.deepcopy(dt_boxes_final)
                triggered_frames.append(img_index)

            tpos[img_index], fpos[img_index], fneg[img_index] = \
                eval_single_image(current_gt, dt_boxes_final)

        tp_total = sum(tpos.values())
        fp_total = sum(fpos.values())
        fn_total = sum(fneg.values())

        f1_score = compute_f1(tp_total, fp_total, fn_total)
        return f1_score, triggered_frames


def load_videostorm_results(filename):
    """Load videostorm result file."""
    videos = []
    perf_list = []
    acc_list = []
    with open(filename, 'r') as f_vs:
        f_vs.readline()
        for line in f_vs:
            line_list = line.strip().split(',')
            videos.append(line_list[0])
            perf_list.append(float(line_list[1]))

            if len(line_list) == 3:
                acc_list.append(float(line_list[2]))

    return videos, perf_list, acc_list


def load_videostorm_profile(filename):
    """Load videostorm profiling file."""
    videos = []
    perf_dict = defaultdict(list)
    acc_dict = defaultdict(list)
    with open(filename, 'r') as f_vs:
        f_vs.readline()  # remove headers
        for line in f_vs:
            line_list = line.strip().split(',')
            video = line_list[0]
            if video not in videos:
                videos.append(video)
            perf_dict[video].append(float(line_list[1]))
            # if len(line_list) == 3:
            acc_dict[video].append(float(line_list[2]))

    return videos, perf_dict, acc_dict
