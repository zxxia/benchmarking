import json
import os
import numpy as np
from collections import defaultdict, Counter


def resol_str_to_int(resol_str):
    """Convert a human readable resolution to integer.

    e.g. "720p" -> 720
    """
    return int(resol_str.strip('p'))


def create_dir(path):
    if not os.path.exists(path):
        print('create path ', path)
        os.makedirs(path)
    else:
        print(path, 'already exists!')


def load_metadata(filename):
    with open(filename) as f:
        metadata = json.load(f)
    return metadata


def compute_f1(tp, fp, fn):
    if tp:
        precison = float(tp) / (tp + fp)
        recall = float(tp) / (tp + fn)
        f1 = 2*(precison*recall)/(precison+recall)
    else:
        if fn:
            f1 = 0
        else:
            f1 = 1
    return f1


def nms(dets, thresh):
    """Perform nms."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def interpolation(point_a, point_b, target_x):
    k = float(point_b[1] - point_a[1])/(point_b[0] - point_a[0])
    b = point_a[1] - point_a[0]*k
    target_y = k*target_x + b
    return target_y


def Most_Common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]


def IoU(boxA, boxB):
    """Return IoU of two boxes."""
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


# def CDF(data, num_bins=20, normed=True):
#     data_size = len(data)
#
#     # Set bins edges
#     data_set = sorted(set(data))
#     bins = np.append(data_set, data_set[-1]+1)
#
#     # Use the histogram function to bin the data
#     counts, bin_edges = np.histogram(data, bins=bins, density=False)
#     counts = counts.astype(float)/data_size
#
#     # Find the cdf
#     cdf = np.cumsum(counts)
#
#     return bin_edges[0:-1], cdf
