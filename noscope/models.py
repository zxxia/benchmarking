import argparse
from data_generator import DataGenerator
import numpy as np
import os
import keras
import time
import tempfile
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten,\
    Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from skimage.io import imread
from skimage.transform import resize
from identify_dominant_class import dominant_classes
from keras.applications.vgg16 import VGG16
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.preprocessing import image
import sklearn
import sklearn.metrics
from collections import Counter
from keras.layers import Convolution2D, MaxPooling2D


partition = {'train': range(1, 18001),
             'val': range(18001, 22001),
             'test': range(22001, 30001)}  # IDs


def build_model(dim, model_arch, all_classes):
    """Build up a simple model(classification)."""
    nb_classes = len(all_classes)
    if model_arch == 'alexnet':
        model = alexnet((*dim, 3), nb_classes)
    # if model_arch == 'vgg16':
    #     model = vgg16(nb_classes)
    elif model_arch == 'mobilenet_v2':
        model = mobilenet_v2(nb_classes)
    elif model_arch == 'noscope':
        model = noscope_model((*dim, 3), nb_classes)
    elif model_arch == 'O1':
        model = O1_model((*dim, 3), nb_classes)
    elif model_arch == 'O2':
        model = O2_model((*dim, 3), nb_classes)
    else:
        print('Invalid model_arch', model_arch)
        model = None

    return model


def get_callbacks(csv_logger, model_fname, patience=5):
    return [
        EarlyStopping(monitor='loss',
                      patience=patience, min_delta=0.00001),
        EarlyStopping(monitor='val_loss',
                      patience=patience + 2, min_delta=0.0001),
        ModelCheckpoint(model_fname, save_best_only=True),
        csv_logger]


def noscope_model(input_shape, nb_classes, nb_dense=128, nb_filters=32,
                  nb_layers=1, kernel_size=(3, 3), stride=(1, 1),
                  regression=False):
    assert nb_layers >= 0
    assert nb_layers <= 3
    model = Sequential()
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='same',
                            input_shape=input_shape,
                            subsample=stride,
                            activation='relu'))

    if nb_layers > 0:
        model.add(Convolution2D(nb_filters, 3, 3, border_mode='same',
                                activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

    if nb_layers > 1:
        model.add(Convolution2D(nb_filters * 2, 3, 3, border_mode='same',
                                activation='relu'))
        model.add(Convolution2D(nb_filters * 2, 3, 3, border_mode='same',
                                activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

    if nb_layers > 2:
        model.add(Convolution2D(nb_filters * 4, 3, 3, border_mode='same',
                                activation='relu'))
        model.add(Convolution2D(nb_filters * 4, 3, 3, border_mode='same',
                                activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(nb_dense, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.RMSprop(lr=0.0001),
                  metrics=['accuracy'])
    return model


def alexnet(input_shape, nb_classes):
    # (3) Create a sequential model
    model = Sequential()

    # 1st Convolutional Layer
    model.add(Conv2D(filters=96, input_shape=input_shape, kernel_size=(11, 11),
                     strides=(4, 4), padding='valid'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
    # Batch Normalisation before passing it to the next layer
    model.add(BatchNormalization())

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(5, 5),
                     strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3, 3),
                     strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 4th Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3, 3),
                     strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 5th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3, 3),
                     strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # Passing it to a dense layer
    model.add(Flatten())
    # 1st Dense Layer
    model.add(Dense(4096))
    model.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.4))
    # Batch Normalisation
    # model.add(BatchNormalization())

    # 2nd Dense Layer
    model.add(Dense(4096))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    # model.add(BatchNormalization())

    # 3rd Dense Layer
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    if nb_classes == 1:
        loss = 'binary_crossentropy'
    else:
        loss = 'categorical_crossentropy'

    sgd = keras.optimizers.SGD(
        lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=loss, optimizer=sgd, metrics=['accuracy'])

    # model.compile(loss=loss, optimizer=keras.optimizers.RMSprop(lr=0.0001),\
    #  metrics=['accuracy'])
    return model


def O1_model(input_shape, nb_classes, lr_mult=1):

    model = Sequential()
    model.add(Conv2D(96, (11, 11), input_shape=input_shape,
                     strides=4, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
    model.add(Conv2D(256, (5, 5), padding='same',
                     strides=2, activation='relu'))

    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

    model.add(Conv2D(384, (3, 3), padding='same', activation='relu'))

    model.add(Conv2D(384, (3, 3), padding='same', activation='relu'))

    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    if nb_classes == 1:
        loss = 'binary_crossentropy'
    else:
        loss = 'categorical_crossentropy'
    model.compile(loss=loss, optimizer=keras.optimizers.Adam(lr=0.0001),
                  metrics=['accuracy'])
    return model


def O2_model(input_shape, nb_classes):

    model = Sequential()
    model.add(Conv2D(64, (11, 11), input_shape=input_shape,
                     strides=4, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
    model.add(Conv2D(256, (5, 5), padding='same', strides=2,
                     activation='relu'))

    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))

    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    if nb_classes == 1:
        loss = 'binary_crossentropy'
    else:
        loss = 'categorical_crossentropy'
    model.compile(loss=loss, optimizer=keras.optimizers.Adam(lr=0.0001),
                  metrics=['accuracy'])

    return model


def mobilenet_v2(nb_classes):
    model = MobileNetV2(weights=None, classes=nb_classes)
    if nb_classes == 1:
        loss = 'binary_crossentropy'
    else:
        loss = 'categorical_crossentropy'
    model.compile(loss=loss, optimizer=keras.optimizers.Adam(lr=0.0001),
                  metrics=['accuracy'])
    return model


def load_label_faster_rcnn(label_file, dominate_classes, start=0, end=30001):
    labels = {}
    with open(label_file, 'r') as f:
        for line in f:
            line_list = line.strip().split(',')
            frame_id = int(line_list[0].replace('.jpg', ''))
            if frame_id < start:
                continue
            if frame_id > end:
                break
            line_list[1] = line_list[1].strip().replace(
                '"', '').replace(' ', '_')
            if line_list[1] not in dominate_classes:
                print(dominate_classes)
                print(line_list[1])
            labels[frame_id] = dominate_classes.index(line_list[1])
        col_count = Counter(list(labels.values()))
        print(col_count)
    return labels


def get_data(dataset, dim):

    path = '/home/zhujun/video_analytics_pipelines/fast/data'
    label_file = os.path.join(
        path, dataset+'_label_from_COCO_direct_filtered.csv')
    # use all the objects that appears in the training dataset
    _, dominate_classes = dominant_classes(label_file, -1,
                                           partition['train'][0],
                                           partition['test'][-1])  # partition['train'][-1]
    nb_classes = len(dominate_classes)
    if nb_classes == 1:
        nb_classes = 2

    params = {'dim': dim,
              'batch_size': 32,
              'n_classes': nb_classes,
              'n_channels': 3,
              'shuffle': True}
    # Datasets
    # labels = load_label(dataset, dominate_classes)
    labels = load_label_faster_rcnn(label_file, dominate_classes,
                                    partition['train'][0],
                                    partition['test'][-1])

    # Generators
    training_generator = DataGenerator(partition['train'],
                                       labels, dataset, **params)
    validation_generator = DataGenerator(partition['val'],
                                         labels, dataset, **params)
    return nb_classes, training_generator, validation_generator, labels

# Use data generator


def run_model(model, logfile_name, training_generator,
              validation_generator, patience=5):
    # Train model on dataset
    temp_fname = tempfile.mkstemp(suffix='.hdf5', dir='/tmp/')[1]
    csv_logger = CSVLogger(logfile_name, append=True, separator=';')
    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        # steps_per_epoch=300,
                        # validation_steps=50,
                        epochs=30,
                        callbacks=get_callbacks(csv_logger,
                                                temp_fname, patience))
    model.load_weights(temp_fname)
    os.remove(temp_fname)
    return


def probas_to_classes(y_pred):
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        return np.argmax(y_pred, axis=1)
    return np.array([1 if p > 0.5 else 0 for p in y_pred])


def predict_labels(model, X_test_gen, batch_size):
    begin = time.time()
    data_len = partition['test'][-1] - partition['test'][0] + 1
    # Alternate way to compute the classes
    proba = model.predict_generator(X_test_gen,
                                    steps=int(data_len / float(batch_size)) +
                                    (data_len % batch_size > 0),
                                    verbose=0)
    predicted_labels = probas_to_classes(proba)
    end = time.time()
    return predicted_labels, end - begin


def load_test_data(dataset, batch_size, dim, n_channels=3):
    path = '/home/zhujun/video_analytics_pipelines/fast/data/' + dataset + '/'
    batch_start = partition['test'][0]
    while batch_start < partition['test'][-1] + 1:
        batch_end = min(batch_start + batch_size, partition['test'][-1] + 1)
        X = np.empty((batch_end-batch_start, *dim, n_channels))
        # Generate data
        list_IDs_temp = range(batch_start, batch_end)
        batch_start = batch_end
        for i, ID in enumerate(list_IDs_temp):
            img = image.load_img(path + format(ID, '06d') + '.jpg',
                                 target_size=dim)
            image.img_to_array(img)
            X[i, ] = (image.img_to_array(img)).astype('float32') / 255
        yield X


def evaluate_model(model, dataset, labels, dim=(224, 224), batch_size=32):
    X_test_gen = load_test_data(dataset, batch_size, dim, n_channels=3)
    predicted_labels, test_time = predict_labels(model,
                                                 X_test_gen, batch_size)

    true_labels = []
    for i in range(partition['test'][0], partition['test'][-1] + 1):
        true_labels.append(labels[i])

    metrics = {'recall': sklearn.metrics.recall_score(true_labels,
                                                      predicted_labels, average='micro'),
               'precision': sklearn.metrics.precision_score(true_labels,
                                                            predicted_labels, average='micro'),
               'accuracy': sklearn.metrics.accuracy_score(true_labels,
                                                          predicted_labels),
               'f1': sklearn.metrics.f1_score(true_labels,
                                              predicted_labels, average='micro'),
               'weighted_f1': sklearn.metrics.f1_score(true_labels,
                                                       predicted_labels, average='weighted'),
               'test_time': test_time}
    return metrics


def main():
    # initialization
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', required=True,
                        help='GPU number')
    parser.add_argument('--dataset', required=True,
                        help='Dataset name')
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    model_arch = args.model

    # config GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    set_session(tf.Session(config=config))

    # create data generators
    if model_arch == 'alexnet' or model_arch == 'O1' or model_arch == 'O2':
        dim = (227, 227)
    else:
        dim = (224, 224)

    nb_classes, training_generator, validation_generator, labels = get_data(
        args.dataset, dim)

    print('number of classes', nb_classes)
    # create model
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
    print(model.summary())

    logfile_name = args.dataset + '_' + model_arch + '_log.csv'
    run_model(model, logfile_name, training_generator, validation_generator)

    metrics = evaluate_model(model, args.dataset, labels, dim)

    print(metrics)

    # serialize model to JSON
    model_json = model.to_json()
    output_path = '/home/zhujun/video_analytics_pipelines/fast/data/' + \
        args.dataset
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(output_path + '/' + model_arch + '_model_best_new_gt_filtered.json',
              'w') as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(output_path + '/' + model_arch +
                       '_model_best_new_gt_filtered.h5')
    print("Saved model to disk")

    return


if __name__ == '__main__':
    main()
