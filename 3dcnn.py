import argparse
import os

import cv2
import matplotlib
matplotlib.use('AGG')

import matplotlib.pyplot as plt
import numpy as np

from keras.layers import (Activation, Conv3D, Dense, Dropout, Flatten,
                          MaxPooling3D, MaxPooling2D)

from keras.layers.advanced_activations import LeakyReLU
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split

from tqdm import tqdm

from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import Input, Dense
import keras
import math


##
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


def plot_history(history, result_dir):
    plt.plot(history.history['acc'], marker='.')
    plt.plot(history.history['val_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig(os.path.join(result_dir, 'model_accuracy.png'))
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'model_loss.png'))
    plt.close()


def save_history(history, result_dir):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, 'result.txt'), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))
    


def get_frames(dir_path, width, height):
    frames_filenames = os.listdir(dir_path)
    frames_filenames = sorted(frames_filenames)  # NO SHUFFLING!
    frames = [cv2.imread(f'{dir_path}/{filename}') for filename in frames_filenames]
    frames = [cv2.resize(frame, (width, height)) for frame in frames]
    return np.array(frames) / 255  # Normalization


def get_label(dir_path, classes):
    *head, class_name, video_name = dir_path.split('/')
    return classes.index(class_name)


def new_generator(subset, classes, width, height, batch_size, n_frames):
    n_steps = len(subset) // batch_size
    for step in range(n_steps):
        batch_directories = subset[step * batch_size: (step + 1) * batch_size]
        inputs = [get_frames(directory, width, height) for directory in batch_directories]
        # min_length = min(sample.shape[0] for sample in inputs)
        inputs = [sample[:n_frames, :, :, :] for sample in inputs]
        outputs = [get_label(directory, classes) for directory in batch_directories]
        #(img_rows, img_cols, frames, channel) for transpose
        #turn output from int to 0,1 using one hot encoded keras
        yield np.array(inputs).transpose((0, 2, 3, 1, 4)), to_categorical(np.array(outputs), len(classes))


def prepare_dataset(root_dir_path, validation_subset=0.2, random_state=7, shuffle=True):
    # Identifying the Directories
    classes = os.listdir(root_dir_path)
    samples = [os.listdir(f'{root_dir_path}/{cls}') for cls in classes]
    samples = [[f'{root_dir_path}/{cls}/{dirname}' for dirname in li] for cls, li in zip(classes, samples)]
    flattened = []  # Will contain full path to every subdirectory containing images.
    for li in samples:
        flattened.extend(li)
    # Randomly Splitting the Dataset
    return train_test_split(flattened, test_size=validation_subset, random_state=random_state, shuffle=shuffle), classes


def main():
    parser = argparse.ArgumentParser(
        description='simple 3D convolution for action recognition')
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--videos', type=str, default='UCF101',
                        help='directory where videos are stored')
    parser.add_argument('--nclass', type=int, default=101)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--color', type=bool, default=False)
    parser.add_argument('--skip', type=bool, default=True)
    parser.add_argument('--depth', type=int, default=10)
    args = parser.parse_args()

    img_rows, img_cols  = 224, 224

    frames  = args.depth
    nb_classes = args.nclass
    nb_batch = args.batch

    channel = 3 if args.color else 1


    # Define model
    input_x = Input(shape = (img_rows, img_cols, frames, channel))

    initial_conv = Conv3D(16, kernel_size= (3, 3, 3), padding='same')(input_x)
    initial_conv = LeakyReLU(alpha=.001)(initial_conv)

    initial_conv = Conv3D(32, kernel_size= (3, 3, 3), padding='same')(initial_conv)
    initial_conv = LeakyReLU(alpha=.001)(initial_conv)

    ###########################
    # PARALLEL 1

    conv1 = Conv3D(16, kernel_size=(1, 1, 1),padding='same')(initial_conv)
    conv1 = LeakyReLU(alpha=.001)(conv1)
    conv1 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv1)

    conv1 = Conv3D(16, kernel_size=(3, 3, 3),padding='same')(conv1)
    conv1 = LeakyReLU(alpha=.001)(conv1)
    
    conv1 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv1)
    conv1 = Conv3D(1, kernel_size=(1, 1, 1),padding='same')(conv1)
    conv1 = LeakyReLU(alpha=.001)(conv1)
    ##############################

    #Parallel 2

    conv2 = Conv3D(8, kernel_size=(1, 1, 1),padding='same')(initial_conv)
    conv2 = LeakyReLU(alpha=.001)(conv2)

    conv2 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv2)
    conv2 = Conv3D(8, kernel_size=(3, 3, 3),padding='same')(conv2)
    conv2 = LeakyReLU(alpha=.001)(conv2)
    
    conv2 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv2)
    conv2 = Conv3D(1, kernel_size=(1, 1, 1),padding='same')(conv2)
    conv2 = LeakyReLU(alpha=.001)(conv2)
    ##############################

    #Parallel 3

    conv3 = Conv3D(4, kernel_size=(1, 1, 1),padding='same')(initial_conv)
    conv3 = LeakyReLU(alpha=.001)(conv3)
    conv3 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv3)

    conv3 = Conv3D(4, kernel_size=(3, 3, 3),padding='same')(conv3)
    conv3 = LeakyReLU(alpha=.001)(conv3)

    conv3 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv3)
    conv3 = Conv3D(1, kernel_size=(1, 1, 1),padding='same')(conv3)
    conv3 = LeakyReLU(alpha=.001)(conv3)
    ###################################

    added = keras.layers.Add()([conv1, conv2, conv3])
    added = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(added)
    
    added = Flatten()(added)

    dense_1 = Dense(784)(added)
    dense_2 = Dense(nb_classes)(dense_1)

    print(dense_2.shape)

    model = Model(input_x, dense_2)
    
    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(), metrics=['categorical_accuracy'])
    
    model.summary() 


    ####################

    # MODEL CHECK POINTS

    filepath="saved_models/dk_3dcnnmodel-{epoch:02d}-{val_acc:.2f}.hd5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

     # GPU CONFIGURATION 
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    set_session(tf.Session(config=config))

    # OLD: print(X_train.shape)
    # OLD: print(Y_train.shape)
    
    # OLD: history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=args.batch,
    #                     epochs=args.epoch, verbose=1, shuffle=True, callbacks=callbacks_list)

    # OLD: history = model.fit_generator(myGenerator(X_train, X_test, Y_train, Y_test, nb_classes, nb_batch), samples_per_epoch = X_train.shape[0], epochs = args.epoch, verbose=1, callbacks=callbacks_list, shuffle = True)

    (train_subset, valid_subset), classes = prepare_dataset(args.videos, validation_subset=0.2, random_state=7, shuffle=True)
    train_steps, valid_steps = len(train_subset) // args.batch, len(valid_subset) // args.batch

    history = model.fit_generator(
        new_generator(train_subset, classes, 224, 224, args.batch, args.depth), shuffle=True,
        epochs=args.epoch, verbose=1, callbacks=callbacks_list, samples_per_epoch=train_steps,
        validation_data=new_generator(valid_subset, classes, 224, 224, args.batch, args.depth), validation_steps=valid_steps,
    )

    # OLD: model.evaluate(X_test, Y_test, verbose=0)

    loss, acc = model.evaluate(new_generator(valid_subset, classes, 224, 224, args.batch, args.depth), steps=valid_steps, verbose=1)
    
    model_json = model.to_json()
    
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    with open(os.path.join(args.output, 'ucf101_3dcnnmodel.json'), 'w') as json_file:
        json_file.write(model_json)
    
    model.save_weights(os.path.join(args.output, 'ucf101_3dcnnmodel-gpu.hd5'))

    # OLD: loss, acc = model.evaluate(X_test, Y_test, verbose=0)
    
    print('Test loss:', loss)
    print('Test accuracy:', acc)

    plot_history(history, args.output)
    save_history(history, args.output)


if __name__ == '__main__':
    main()