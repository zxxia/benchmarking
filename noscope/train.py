import argparse
from collections import Counter, defaultdict
import numpy as np
import os
import keras
import time
import random
import glob
import tempfile
from keras.models import model_from_json, Model, load_model
from keras.layers import Dropout, Dense
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import CSVLogger
from skimage.io import imread
from skimage.transform import resize
# from identify_dominant_class import dominant_classes
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from models import get_callbacks, build_model
from sklearn.metrics import f1_score, confusion_matrix, classification_report,\
    accuracy_score
from shutil import copyfile, rmtree
import tensorflow as tf
from my_utils import DataGenerator
from scipy import spatial


random.seed(0)
np.random.seed(0)


batch_size = 256
# all_classes = ['car','person','truck','bicycle','bus','no_object']
# all_classes = ['potted_plant', 'car','person','truck','bicycle', 'bus',
# 'train', 'motorcycle', 'no_object']


partition_test = (18001, 32000)
partition_train = (1, 18001)
partition_val = (18001, 22001)

short_video_length = 30
frame_rate = 30


def build_train_val(dim, dataset, filename):
    # load label file, create training, validation data generator
    path = '/home/zhujunxiao/video_analytics_pipelines/fast/data/'
    label_file = os.path.join(path,
                              dataset + '_ground_truth' + filename + '.csv')

    # path to cropped images
    img_path = '/home/zhujunxiao/video_analytics_pipelines/fast/data/' + dataset + '/360p/'
    labels = {}
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
    with open(label_file, 'r') as f:
        for line in f:
            line_list = line.strip().split(',')
            current_label = line_list[1].strip().replace(
                '"', '').replace(' ', '_')
            labels[int(line_list[0])] = current_label
            if current_label not in all_classes:
                all_classes.append(current_label)

    # get train images path
    for img_index in range(partition_train[0], partition_train[1]):
        label = labels[img_index]
        assert label in all_classes, print(label)
        train_test[label] += 1
        image_name = img_path + '/' + format(img_index, '06d') + '.jpg'
        train_images.append(image_name)
        basename = os.path.basename(image_name)
        train_labels[basename] = all_classes.index(label)

    # to address training data imbalance problem, reweight those classes
    train_class_weight = {}
    for key, value in train_test.items():
        train_class_weight[all_classes.index(key)] = 1./value * 100.

    # get val images path
    for img_index in range(partition_val[0], partition_val[1]):
        label = labels[img_index]
        assert label in all_classes, print(label)
        val_test[label] += 1
        image_name = img_path + '/' + format(img_index, '06d') + '.jpg'
        val_images.append(image_name)
        basename = os.path.basename(image_name)
        val_labels[basename] = all_classes.index(label)

    print('Train data', len(train_images))
    print(train_test)
    train_distribution = [train_test[key] for key in all_classes]
    print('Val data', len(val_images))
    print(val_test)
    nb_classes = len(all_classes)

    train_gen = DataGenerator(train_images, train_labels, batch_size, dim,
                              n_classes=nb_classes, shuffle=True)
    val_gen = DataGenerator(val_images, val_labels, batch_size, dim,
                            n_classes=nb_classes, shuffle=False)

    return train_gen, val_gen, train_class_weight, train_distribution, \
        all_classes


def parse_args():
    '''Parse arguments.'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', required=True, help='GPU number')
    parser.add_argument('--model', default='O1')
    parser.add_argument('--dataset', required=True, help='Dataset name')
    args = parser.parse_args()
    return args


def main():
    """Implement training NoScope small model.

    Use the partition_train (equivalent to 10min video) to train the small
    model.
    """
    args = parse_args()
    # # config GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    set_session(tf.Session(config=config))
    model_arch = args.model
    dataset = args.dataset
    # select small model structure
    if model_arch == 'alexnet' or model_arch == 'O1' or model_arch == 'O2':
        dim = (227, 227)
    else:
        dim = (224, 224)

    # filename is the suffix of label file. Currently, car and truck are
    # considered as two separate labels.
    filename = '_car_truck_separate'
    train_gen, val_gen, train_class_weight, train_distribution, all_classes = \
        build_train_val(dim, dataset, filename)
    model = build_model(dim, model_arch, all_classes)

    # create temp log file
    logfile_name = 'Test_NS_log.csv'
    temp_fname = tempfile.mkstemp(suffix='.hdf5', dir='/tmp/')[1]
    csv_logger = CSVLogger(logfile_name, append=True, separator=';')
    # start model training
    model.fit_generator(
        train_gen,
        epochs=50,
        validation_data=val_gen,
        class_weight=train_class_weight,
        callbacks=get_callbacks(csv_logger, temp_fname, patience=5))

    # save the trained model
    model.load_weights(temp_fname)
    os.remove(temp_fname)
    model_path = '/home/zhujunxiao/video_analytics_pipelines/final_code/fast/models/noscope/'
    model.save(model_path + dataset + filename + '.h5')

    # nb_classes = len(all_classes)
    # for dataset in ['cropped_crossroad4']:
    # 	test_images, test_labels = build_test(dim, dataset, all_classes)
    # 	num_of_seg = len(test_images) // (frame_rate*short_video_length)
    # 	for seg_index in range(num_of_seg):
    # 		current_test_images = test_images[seg_index*(frame_rate*short_video_length):
    # 										(seg_index+1)*(frame_rate*short_video_length)]

    # 		current_test_labels = {os.path.basename(k): test_labels[os.path.basename(k)] for k in current_test_images}

    # 		test_cn = defaultdict(int)
    # 		for img in current_test_labels.keys():
    # 			test_cn[current_test_labels[img]] += 1
    # 		test_distribution = [test_cn[all_classes.index(key)] for key in all_classes]
    # 		print(train_distribution, test_distribution)
    # 		train_test_dist = 1 - spatial.distance.cosine(train_distribution,
    # 													  test_distribution)

    # 		test_gen = DataGenerator(current_test_images, current_test_labels, batch_size, dim,
    # 		n_classes=nb_classes, shuffle=False)
    # 		_, acc = model.evaluate_generator(
    # 			test_gen)
    # 		# print('test acc: %f' % acc)
    # 		f1_list = []
    # 		avg_gpu_list = []
    # 		full_model_cn_list = []
    # 		for thresh in np.arange(0, 1, 0.1):
    # 			print('Thresh:', thresh)
    # 			y_pred, y_true, y_pipeline, test_time, full_model_cn = get_labels(model,
    # 				current_test_images, current_test_labels, thresh, dim)
    # 			# print('Small model performance:')
    # 			# print('F1 score:', f1_score(y_true, y_pred,average='micro'))
    # 			# print(confusion_matrix(y_true, y_pred))
    # 			# print(classification_report(y_true, y_pred))

    # 			print('Pipeline performance:')
    # 			# print('number of full model:', full_model_cn)
    # 			f1 = f1_score(y_true, y_pipeline, average='micro')
    # 			if thresh == 0:
    # 				assert abs(acc-f1) < 0.01

    # 			f1_list.append(f1)
    # 			print('F1 score:', f1)
    # 			print(confusion_matrix(y_true, y_pipeline))
    # 			print(classification_report(y_true, y_pipeline))
    # 			full_model_speed = 0.1
    # 			small_model_speed = 0.005
    # 			avg_gpu = (full_model_cn * full_model_speed + (len(y_pred) -
    # 			 full_model_cn) * small_model_speed) / len(y_pred)
    # 			avg_gpu_list.append(avg_gpu)
    # 			full_model_cn_list.append(full_model_cn)
    # 		print(list(zip(avg_gpu_list, f1_list, full_model_cn_list)))
    # 		result_file.write(dataset + '_' + str(20 + seg_index) + ',')
    # 		result_file.write(' '.join([str(x) for x in f1_list]) + ',')
    # 		result_file.write(' '.join([str(x) for x in avg_gpu_list]) + ',')
    # 		result_file.write(' '.join([str(x) for x in full_model_cn_list]) + ',')
    # 		result_file.write(' '.join([str(x) for x in train_distribution]) + ',')
    # 		result_file.write(' '.join([str(x) for x in test_distribution]) + ',')
    # 		result_file.write(str(train_test_dist) + '\n')
    return


if __name__ == '__main__':
    main()
