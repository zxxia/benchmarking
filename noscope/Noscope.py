import csv
from collections import defaultdict
from utils.utils import interpolation


class Noscope():
    def __init__(self, thresh_list, profile_file, target_f1=0.9):
        self.thresh_list = thresh_list
        self.target_f1 = target_f1
        self.writer = csv.writer(open(profile_file, 'w', 1))

    def profile(self, video_name, small_model_result, gt, start_frame,
                end_frame):
        f1_list = []
        gpu_list = []
        for thresh in self.thresh_list:
            gpu, acc = self.evaluate(video_name, small_model_result, gt,
                                     start_frame, end_frame, thresh)
            f1_list.append(acc)
            gpu_list.append(gpu)
            self.writer.writerow([video_name, thresh, gpu, acc])

        if f1_list[-1] < self.target_f1:
            target_thresh = None
            target_thresh = self.thresh_list[-1]
        else:
            index = next(x[0] for x in enumerate(f1_list)
                         if x[1] > self.target_f1)
            if index == 0:
                target_thresh = self.thresh_list[0]
            else:
                point_a = (f1_list[index-1], self.thresh_list[index-1])
                point_b = (f1_list[index], self.thresh_list[index])
                target_thresh = interpolation(
                    point_a, point_b, self.target_f1)

        return target_thresh

    def evaluate(self, video_name, small_model_result, gt, start_frame,
                 end_frame, thresh):
        small_model_speed = 2.6
        full_model_speed = 100
        full_model_cn = 0
        tp = 0
        total_cn = 0
        for frame_index in range(start_frame, end_frame):
            confidence_score = small_model_result[frame_index][2]
            y_small_model = small_model_result[frame_index][1]

            if confidence_score < thresh:
                # get full model results as pipeline's result
                y_pipeline = gt[frame_index][0]
                full_model_cn += 1
            else:
                # if confidence score is high, use small model output
                y_pipeline = y_small_model
            total_cn += 1
            if y_pipeline == gt[frame_index][0]:
                tp += 1

        acc = tp/total_cn
        # compute relative gpu time
        gpu = (small_model_speed*(total_cn-full_model_cn) /
               full_model_speed + full_model_cn) / total_cn

        return gpu, acc


def load_ground_truth(ground_truth_file):
    all_classes = ['car', 'person', 'truck',
                   'bicycle', 'bus', 'motorcycle', 'no_object']
    gt = {}
    with open(ground_truth_file, 'r') as f:
        for line in f:
            line_list = line.strip().split(',')
            frame_index = int(line_list[0])
            label = line_list[1]
            assert label in all_classes, print(label, line)
            gt[frame_index] = [label]
            if len(line_list) > 2:
                area = float(line_list[2])
                confidence = float(line_list[3])
                gt[frame_index] = [label, area, confidence]
    return gt


def load_noscope_results(filename):
    """Load videostorm result file."""
    videos = []
    perf_list = []
    acc_list = []
    with open(filename, 'r') as f_vs:
        f_vs.readline()
        for line in f_vs:
            line_list = line.strip().split(',')
            videos.append(line_list[0])
            perf_list.append(float(line_list[2]))
            acc_list.append(float(line_list[3]))

    return videos, perf_list, acc_list


def load_simple_model_classification(result_file):
    """Load predictions from small model."""
    small_model_result = {}
    with open(result_file, 'r') as f:
        f.readline()
        for line in f:
            line_list = line.strip().split(',')
            frame_index = int(line_list[0])
            y_true = line_list[1]
            y_small_model = line_list[2]
            # confidence score
            confidence_score = float(line_list[3])
            small_model_result[frame_index] = (y_true, y_small_model,
                                               confidence_score)

    return small_model_result


# def load_noscope_perf(perf_file, gt, dataset, short_video_length=30, frame_rate=30, thresh=0.8):
#     """
#     perf_file: this file has the prediction of each frame from small model, e.g.,
#                         noscope_small_model_predicted_cropped_crossroad4_2_car_truck_separate.csv
#     gt: ground truth label for each frame
#     dataset: dataset name, e.g., 'cropped_crossroad4_2'
#     short_video_length: 30 seconds
#     """
#     tp_cn = defaultdict(int)
#     full_model_cn = defaultdict(int)
#     total_cn = defaultdict(int)
#     acc_pipeline = {}
#     gpu = {}
#     # small_model_result = {}
#     small_model_speed = 2.6
#     full_model_speed = 100
#
#     small_model_result = load_simple_model_classification(perf_file)
#
#     # thresh = 0.8
#     # threshold for triggering full model, if small model's confidence
#     # score is higher than thresh, use small model results, otherwise,
#     # trigger full model
#     # for frame_index in sorted(small_model_result.keys()):
#     # seg_index = frame_index//(short_video_length*frame_rate)
#     for frame_index in range(start, end):
#         confidence_score = small_model_result[frame_index][2]
#         y_small_model = small_model_result[frame_index][1]
#
#         if confidence_score < thresh:
#             # get full model results as pipeline's result
#             y_pipeline = gt[frame_index][0]
#             full_model_cn[seg_index] += 1
#         else:
#             # if confidence score is high, use small model output
#             y_pipeline = y_small_model
#         total_cn[seg_index] += 1
#         if y_pipeline == gt[frame_index][0]:
#             tp_cn[seg_index] += 1
#
#     for key in tp_cn.keys():
#         seg_name = dataset + '_' + str(key)
#         acc_pipeline[seg_name] = tp_cn[key]/total_cn[key]
#         # compute relative gpu time
#         gpu[seg_name] = (small_model_speed*(total_cn[key]-full_model_cn[key]) /
#                          full_model_speed + full_model_cn[key])/total_cn[key]
#
#     return acc_pipeline, gpu
