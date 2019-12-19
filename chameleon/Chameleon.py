from collections import defaultdict


class Chameleon:
    def __init__(self, model_list, target_f1=0.9):
        self.model_list = model_list
        self.target_f1 = target_f1

    def profile(self, video_dict, gt, frame_range):
        for model in self.model_list:
            pred_labels = [video_dict[i]
                           for i in range(frame_range[0], frame_range[1] + 1)]
            gt_labels = [gt[i][0]
                         for i in range(frame_range[0], frame_range[1] + 1)]
            f1 = compute_f1(pred_labels, gt_labels)
            print(f1)


def parse_model_predictions(filename):
    mobilenet_pred = {}
    inception_pred = {}
    resnet50_pred = {}
    with open(filename, 'r') as f:
        for line in f:
            cols = line.strip().split(',')
            img_idx = int(cols[0])
            mobilenet_pred[img_idx] = cols[2]
            inception_pred[img_idx] = cols[3]
            resnet50_pred[img_idx] = cols[4]
    return mobilenet_pred, inception_pred, resnet50_pred


def load_ground_truth(ground_truth_file):
    gt = {}
    all_classes = ['car', 'person', 'truck',
                   'bicycle', 'bus', 'motorcycle', 'no_object']
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


def compute_f1(pred_labels, gt_labels):
    assert len(pred_labels) == len(gt_labels)
    tp = 0
    cnt = 0
    for pred, gt in zip(pred_labels, gt_labels):
        if pred == gt:
            tp += 1
        cnt += 1
    print(tp/cnt)

# def load_model_selection_perf(perf_file, dataset, gt, short_video_length=30,
#                               frame_rate=30):
#     # load prediction results and compute accuracy (avg. f1 score)
#
#     tp = defaultdict(int)
#     total_cn = defaultdict(int)
#     acc = {}
#     acc_per_seg = defaultdict(list)
#     with open(perf_file, 'r') as f:
#         for line in f:
#             line_list = line.strip().split(',')
#             img_index = int(line_list[0])
#             seg_index = img_index//(short_video_length*frame_rate)
#             label = line_list[2]
#             key = ('mobilenet', dataset + '_' + str(seg_index))
#             if label == gt[img_index][0]:
#                 tp[key] += 1
#             total_cn[key] += 1
#
#             label = line_list[3]
#             key = ('inception', dataset + '_' + str(seg_index))
#             if label == gt[img_index][0]:
#                 tp[key] += 1
#             total_cn[key] += 1
#
#             label = line_list[4]
#             key = ('resnet50', dataset + '_' + str(seg_index))
#             if label == gt[img_index][0]:
#                 tp[key] += 1
#             total_cn[key] += 1
#
#     seg_name = []
#     for key in sorted(tp.keys()):
#         acc[key] = tp[key]/total_cn[key]
#         if key[1] not in seg_name:
#             seg_name.append(key[1])
#
#     for seg in seg_name:
#         for model in ['mobilenet', 'inception', 'resnet50']:
#             acc_per_seg[seg].append(acc[(model, seg)])
#         acc_per_seg[seg].append(1)
#
#     return acc, acc_per_seg
