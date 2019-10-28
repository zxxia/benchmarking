# import keras
import json
import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
# from keras.preprocessing import image
from collections import defaultdict, Counter


def resol_str_to_int(resol_str):
    """ convert a human readable resolution to integer
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
# def nms (boxes,overlap):

#       trial = np.zeros((len(boxes),5))
#       trial[:] = boxes[:]
#       x1 = trial[:,0]
#       y1 = trial[:,1]
#       x2 = trial[:,2]
#       y2 = trial[:,3]
#       score = trial[:,4]
#       area = (x2-x1+1)*(y2-y1+1)

#       #vals = sort(score)
#       I = np.argsort(score)
#       pick = []
#       count = 1
#       while (I.size!=0):
#               #print "Iteration:",count
#               last = I.size
#               i = I[last-1]
#               pick.append(i)
#               suppress = [last-1]
#               for pos in range(last-1):
#                       j = I[pos]
#                       xx1 = max(x1[i],x1[j])
#                       yy1 = max(y1[i],y1[j])
#                       xx2 = min(x2[i],x2[j])
#                       yy2 = min(y2[i],y2[j])
#                       w = xx2-xx1+1
#                       h = yy2-yy1+1
#                       if (w>0 and h>0):
#                               o = w*h/area[j]
#                               print('overlap', o)
#                               if (o >overlap):
#                                       suppress.append(pos)
#                                       print('suppress', (i,j))
#               I = np.delete(I,suppress)
#               count = count + 1
#       return pick

# def IoU(boxA, boxB):
#       xA = max(boxA[0], boxB[0])
#       yA = max(boxA[1], boxB[1])
#       xB = min(boxA[2], boxB[2])
#       yB = min(boxA[3], boxB[3])

#   # compute the area of intersection rectangle
#       interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

#   # compute the area of both the prediction and ground-truth
#   # rectangles
#       boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
#       return float(interArea)/boxBArea


def interpolation(point_a, point_b, target_x):
    k = float(point_b[1] - point_a[1])/(point_b[0] - point_a[0])
    b = point_a[1] - point_a[0]*k
    target_y = k*target_x + b
    return target_y

def Most_Common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]


def IoU(boxA, boxB):
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


def CDF(data, num_bins=20, normed=True):
    data_size=len(data)

    # Set bins edges
    data_set = sorted(set(data))
    bins = np.append(data_set, data_set[-1]+1)

    # Use the histogram function to bin the data
    counts, bin_edges = np.histogram(data, bins=bins, density=False)
    counts = counts.astype(float)/data_size

    # Find the cdf
    cdf = np.cumsum(counts)

    return bin_edges[0:-1], cdf


# class DataGenerator(keras.utils.Sequence):
#     'Generates data for Keras'
#     def __init__(self, image_list, labels, batch_size=128, dim=(224, 224),
#                  n_channels=3, n_classes=10, shuffle=True):
#         'Initialization'
#         self.dim = dim
#         self.batch_size = batch_size
#         self.labels = labels
#         self.image_list = image_list
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.shuffle = shuffle
#         self.on_epoch_end()
#
#     def __len__(self):
#         'Denotes the number of batches per epoch'
#         return int(np.floor(len(self.image_list) / self.batch_size))
#
#     def __getitem__(self, index):
#         'Generate one batch of data'
#         # Generate indexes of the batch
#         indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
#         # Find list of IDs
#         images_temp = [self.image_list[k] for k in indexes]
#         # Generate data
#         X, y = self.__data_generation(images_temp)
#
#         return X, y
#
#     def on_epoch_end(self):
#         'Updates indexes after each epoch'
#         self.indexes = np.arange(len(self.image_list))
#         if self.shuffle:
#             np.random.shuffle(self.indexes)
#
#     def __data_generation(self, images_temp):
#         'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
#         # Initialization
#         X = np.empty((self.batch_size, *self.dim, self.n_channels))
#         y = np.empty((self.batch_size), dtype=int)
#
#         # Generate data
#         for i, img_filename in enumerate(images_temp):
#             # Store sample
#             img = image.load_img(img_filename, target_size=self.dim)
#             # image.img_to_array(img)
#             X[i,] = (image.img_to_array(img)).astype('float32') / 255
#             basename = os.path.basename(img_filename)
#             y[i] = self.labels[basename]
#
#         return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
#
