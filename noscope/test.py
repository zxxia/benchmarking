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
from noscope_alexnet import get_callbacks, noscope_model, alexnet, O1_model, \
    O2_model, mobilenet_v2
from sklearn.metrics import f1_score, confusion_matrix, classification_report,\
    accuracy_score
from shutil import copyfile, rmtree
import tensorflow as tf
from my_utils import DataGenerator
from scipy import spatial


# same as the setting in training code
random.seed(0)
np.random.seed(0)


batch_size = 256


partition_test = (18001, 32000)
partition_train = (1, 18001)
partition_val = (18001, 22001)

short_video_length = 30
frame_rate = 30


# same as the function in training code
def build_model(dim, model_arch):
    nb_classes = len(all_classes)
    if model_arch == 'alexnet':
        model = alexnet((*dim, 3), nb_classes)
    if model_arch == 'vgg16':
        model = vgg16(nb_classes)
    if model_arch == 'mobilenet_v2':
        model = mobilenet_v2(nb_classes)
    if model_arch == 'noscope':
        model = noscope_model((*dim, 3), nb_classes)
    if model_arch == 'O1':
        model = O1_model((*dim, 3), nb_classes)
    if model_arch == 'O2':
        model = O2_model((*dim, 3), nb_classes)

    return model


def get_labels(model, test_images, test_labels, dim):

    def test_generator(test_images, dim):

        X = np.empty((batch_size, *dim, 3))
        i = 0
        batch_cn = 0
        for img_filename in test_images:
            # load, pre-process each test image for model prediction
            img = load_img(img_filename,
                           target_size=dim)
            X[i, ] = (img_to_array(img)).astype('float32') / 255
            i += 1
            if i == batch_size:
                batch_cn += 1
                yield X
                X = np.empty((batch_size, *dim, 3))
                i = 0

    begin = time.time()
    predictions = model.predict_generator(test_generator(test_images, dim),
                                          steps=len(test_images)//batch_size)
    end = time.time()
    print('testing time for {} images is {}'.format(len(test_images), end-begin))
    y_true = []
    cn = 0
    all_filenames = []

    # load true labels for test images
    for img_filename in test_images:
        if cn >= len(predictions):
            break
        cn += 1
        basename = os.path.basename(img_filename)
        y_true.append(test_labels[basename])
        all_filenames.append(int(basename.replace('.jpg', '')))

    # prediction label is the index of highest confidence score
    y_pred = np.empty(len(predictions), dtype=int)
    assert len(predictions) == len(y_true)
    result = {}
    for i in range(len(predictions)):
        y_pred = np.argmax(predictions[i])

        key = all_filenames[i]
        result[key] = (y_true[i], y_pred, max(predictions[i]))

    return result


def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', required=True, help='GPU number')
    parser.add_argument('--model', default='O1', help='Small model structure')
    parser.add_argument('--test', required=True, help='Training dataset name')
    parser.add_argument('--train', required=True, help='Test dataset name')
    args = parser.parse_args()
    # # config GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    set_session(tf.Session(config=config))
    return args


def build_test(dim, dataset, filename, train):
    # similar to load training data
    path = '/home/zhujunxiao/video_analytics_pipelines/fast/data/'
    label_file = os.path.join(path,
                              dataset + '_ground_truth' + filename + '.csv')
    train_label_file = os.path.join(path,
                                    train + '_ground_truth' + filename + '.csv')

    img_path = '/home/zhujunxiao/video_analytics_pipelines/fast/data/' + dataset + '/360p/'
    labels = {}
    test_images = []
    test_labels = {}
    test_test = defaultdict(int)

    # identify all the classes that appear in the training data, so later the label
    # can be mapped to label indices
    all_classes = []
    with open(train_label_file, 'r') as f:
        for line in f:
            line_list = line.strip().split(',')
            current_label = line_list[1].strip().replace(
                '"', '').replace(' ', '_')
            if current_label not in all_classes:
                all_classes.append(current_label)
    print(all_classes)

    # load test data label
    with open(label_file, 'r') as f:
        for line in f:
            line_list = line.strip().split(',')
            current_label = line_list[1].strip().replace(
                '"', '').replace(' ', '_')
            labels[int(line_list[0])] = current_label
            # if current_label not in all_classes:
            # 	all_classes.append(current_label)

    # test on all images, now 32000 is a hard-coded number, you can change it to
    # total frame count

    for img_index in range(1, 32000):
        label = labels[img_index]
        # label that appears in the test data
        assert label in all_classes, print(label)
        # should also appear in the training data
        test_test[label] += 1
        image_name = img_path + '/' + format(img_index, '06d') + '.jpg'
        test_images.append(image_name)
        basename = os.path.basename(image_name)
        test_labels[basename] = all_classes.index(label)

    print('test data', len(test_images))
    print(test_test)
    return test_images, test_labels, all_classes


def main():
    ''' Implementation of get NoScope small model output (label and confidence
    score on each image)
    '''

    args = setup()

    # Important! need to specify what dataset is used for training, and what
    # dataset is used for test
    train = args.train
    dataset = args.test

    # filename is the suffix of label file. Currently, car and truck are considered
    # as two separate labels.
    filename = '_car_truck_separate'
    if model_arch == 'alexnet' or model_arch == 'O1' or model_arch == 'O2':
        dim = (227, 227)
    else:
        dim = (224, 224)

    # load the previously trained small model
    model_path = '/home/zhujunxiao/video_analytics_pipelines/final_code/fast/models/noscope/'
    model = load_model(model_path + train + filename + '.h5')

    # result file, record the predicted label and confidence score of each frame
    predicted_label_file = 'noscope_small_model_predicted_' + dataset + filename + '.csv'

    # load test data, and get test accuracy
    test_images, test_labels, all_classes = build_test(
        dim, dataset, filename, train)
    nb_classes = len(all_classes)
    test_gen = DataGenerator(test_images, test_labels, batch_size, dim,
                             n_classes=nb_classes, shuffle=False)
    _, acc = model.evaluate_generator(test_gen)

    # write prediction labels and confidence scores for future evaluation
    result = get_labels(model, test_images, test_labels, dim)
    with open(predicted_label_file, 'w') as f:
        f.write('frame_index, y_true, y_predict, confidence score\n')
        for key in sorted(result.keys()):
            f.write(str(key) + ',' + all_classes[result[key][0]] + ',' +
                    all_classes[result[key][1]] + ',' + str(result[key][2]) + '\n')

    return


if __name__ == '__main__':
    main()
