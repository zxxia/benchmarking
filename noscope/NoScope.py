"""NoScope Implementation."""
import csv
# from collections import defaultdict
from benchmarking.utils.utils import interpolation
import argparse
from collections import defaultdict
import os
import random
import tempfile
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import CSVLogger
from models import get_callbacks, build_model
from my_utils import DataGenerator

class NoScope():
    """NoScope definition."""

    def __init__(self, thresh_list, profile_file, target_f1=0.9):
        """Constructor."""
        self.thresh_list = thresh_list
        self.target_f1 = target_f1
        self.writer = csv.writer(open(profile_file, 'w', 1))

    def profile(self, video_name, small_model_result, gt, start_frame,
                end_frame):
        """Profile a list of confidence score thresholds."""
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
        """Evaluate the target confidence score thresholds."""
        # TODO: Model spped is not consistent with other pipelines
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


def load_groundtruth(ground_truth_file):
    all_classes = ['car', 'person', 'truck',
                   'bicycle', 'bus', 'motorcycle', 'no_object']
    gt = {}
    with open(ground_truth_file, 'r') as f:
        frame_index = int(line_list[0])
        for line in f:
            line_list = line.strip().split(',')
            if len(line_list) > 2:
                # label = line_list[1]
                assert label in all_classes, print(label, line)
                gt[frame_index] = [label]
                area = float(line_list[2])
                confidence = float(line_list[3])
                gt[frame_index] = [label, area, confidence]
            else:
                gt[frame_index] = []
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



def train_small_model(dataset, gpu_num, model_save_path, video,
                      model_arch = 'O1', num_of_epochs=30, batch_size=64):
    # # config GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    set_session(tf.Session(config=config))
    # select small model structure
    if model_arch == 'alexnet' or model_arch == 'O1' or model_arch == 'O2':
        dim = (227, 227)
    else:
        dim = (224, 224)

    # filename is the suffix of label file. Currently, car and truck are
    # considered as two separate labels.
    train_gen, val_gen, train_class_weight, train_distribution, all_classes = \
        build_train_val(dim, dataset, video,batch_size)
    model = build_model(dim, model_arch, all_classes)


    # create temp log file
    logfile_name = 'Test_NS_log.csv'
    temp_fname = tempfile.mkstemp(suffix='.hdf5', dir='/tmp/')[1]
    csv_logger = CSVLogger(logfile_name, append=True, separator=';')
    # start model training
    model.fit_generator(
        train_gen,
        epochs=num_of_epochs,
        validation_data=val_gen,
        class_weight=train_class_weight,
        callbacks=get_callbacks(csv_logger, temp_fname, patience=5))

    # save the trained model
    model.load_weights(temp_fname)
    os.remove(temp_fname)
    model.save(os.path.join(model_save_path, dataset + '.h5'))
    return


def build_train_val(dim, dataset, video, batch_size):
    """Build train-validatation set."""
    # load label file, create training, validation data generator

    partition_train = (1, 18001)
    partition_val = (18001, 22001)



    train_images = []
    train_labels = {}
    val_images = []
    val_labels = {}
    train_test = defaultdict(int)
    val_test = defaultdict(int)
    all_classes = []


    # identify all the classes that appear in the training data,
    # so later the label
    # can be mapped to label indices
    labels = video.get_video_classification_label()
    all_classes = list(set([x[0] for x in labels.values()]))
    print(all_classes)

    # get train images path
    for img_index in range(partition_train[0], partition_train[1]):
        label = labels[img_index][0]
        train_test[label] += 1
        image_name = video.get_frame_image_name(img_index)
        train_images.append(image_name)
        basename = os.path.basename(image_name)
        train_labels[basename] = all_classes.index(label)

    # to address training data imbalance problem, reweight those classes
    train_class_weight = {}
    for key, value in train_test.items():
        train_class_weight[all_classes.index(key)] = 1./value * 100.

    # get val images path
    for img_index in range(partition_val[0], partition_val[1]):
        label = labels[img_index][0]
        assert label in all_classes, print(label)
        val_test[label] += 1
        image_name = video.get_frame_image_name(img_index)
        val_images.append(image_name)
        basename = os.path.basename(image_name)
        val_labels[basename] = all_classes.index(label)

    print('Train data', len(train_images))
    print(train_test)
    train_distribution = [train_test[key] for key in all_classes]
    print('Val data', len(val_images))
    print(val_test)
    nb_classes = len(all_classes)
    print(nb_classes)
    train_gen = DataGenerator(train_images, train_labels, batch_size, dim,
                              n_classes=nb_classes, shuffle=True)
    val_gen = DataGenerator(val_images, val_labels, batch_size, dim,
                            n_classes=nb_classes, shuffle=False)

    return train_gen, val_gen, train_class_weight, train_distribution, \
        all_classes

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
