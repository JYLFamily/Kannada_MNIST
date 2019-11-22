# coding:utf-8

import os
import gc
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from keras.utils import Sequence
from keras.models import Sequential
from keras.applications import VGG16
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)


class FitGenerator(Sequence):
    def __init__(self, feature, label, shuffle, batch_size, image_augment):
        self.__index = np.arange(feature.shape[0])
        self.__feature, self.__label = feature, label
        self.__shuffle, self.__batch_size, self.__image_augment = shuffle, batch_size, image_augment

    def __len__(self):
        return self.__feature.shape[0] // self.__batch_size

    def __getitem__(self, idx):
        index = self.__index[idx * self.__batch_size: (idx + 1) * self.__batch_size]
        batch_feature, batch_label = (
            np.array([zoom(image, zoom=(8, 8, 1)) / 255 for image in self.__feature[index]]), self.__label[index])

        if self.__image_augment is not None:
            batch_feature, batch_label = (
                next(self.__image_augment.flow(np.array(batch_feature), batch_label, batch_size=self.__batch_size)))

        return np.repeat(batch_feature, 3, -1), batch_label

    def on_epoch_end(self):
        if self.__shuffle:
            np.random.shuffle(self.__index)


class PredictGenerator(Sequence):
    def __init__(self, feature):
        self.__index = np.arange(feature.shape[0])
        self.__feature = feature

    def __len__(self):
        return self.__feature.shape[0]

    def __getitem__(self, idx):
        index = self.__index[idx: (idx + 1)]

        batch_feature = np.array([zoom(image, zoom=(8, 8, 1)) / 255 for image in self.__feature[index]])

        return np.repeat(batch_feature, 3, -1)


class VggNetPretrain(object):
    def __init__(self, *, path):
        self.__path = path
        self.__train, self.__valid, self.__test = [None for _ in range(3)]
        self.__train_feature, self.__valid_feature, self.__test_feature = [None for _ in range(3)]
        self.__train_label, self.__valid_label, self.__test_index = [None for _ in range(3)]

        self.__image_data_generator = None
        self.__vgg_net = None

    def data_read(self):
        self.__train = pd.read_csv(os.path.join(self.__path, "train.csv"))
        self.__valid = pd.read_csv(os.path.join(self.__path, "Dig-MNIST.csv"))
        self.__test = pd.read_csv(os.path.join(self.__path, "test.csv"))

    def data_prepare(self):
        self.__train_feature, self.__train_label = (
            self.__train.iloc[:, 1:].copy(deep=True), self.__train.iloc[:, 0].copy(deep=True))
        self.__valid_feature, self.__valid_label = (
            self.__valid.iloc[:, 1:].copy(deep=True), self.__valid.iloc[:, 0].copy(deep=True))
        self.__test_feature, self.__test_index = (
            self.__test.iloc[:, 1:].copy(deep=True), self.__test.iloc[:, [0]].copy(deep=True))
        del self.__train, self.__valid, self.__test
        gc.collect()

        self.__train_feature, self.__train_label = self.__train_feature.to_numpy(), self.__train_label.to_numpy()
        self.__valid_feature, self.__valid_label = self.__valid_feature.to_numpy(), self.__valid_label.to_numpy()
        self.__test_feature = self.__test_feature.to_numpy()

        self.__train_feature = self.__train_feature.reshape((-1, 28, 28, 1))
        self.__valid_feature = self.__valid_feature.reshape((-1, 28, 28, 1))
        self.__test_feature = self.__test_feature.reshape((-1, 28, 28, 1))

        self.__image_data_generator = ImageDataGenerator(
            rotation_range=15,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
            data_format="channels_last"
        )

    def model_fit_predict(self):
        # net
        conv_base = VGG16(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
        conv_base.trainable = False

        self.__vgg_net = Sequential()
        self.__vgg_net.add(conv_base)
        self.__vgg_net.add(Flatten())
        self.__vgg_net.add(Dense(units=4096, activation="relu"))
        self.__vgg_net.add(Dropout(0.5))
        self.__vgg_net.add(Dense(units=4096, activation="relu"))
        self.__vgg_net.add(Dropout(0.5))
        self.__vgg_net.add(Dense(units=10, activation="softmax"))

        # net fit
        self.__vgg_net.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        self.__vgg_net.fit_generator(
            generator=FitGenerator(self.__train_feature, self.__train_label, True, 256, self.__image_data_generator),
            steps_per_epoch=self.__train_feature.shape[0] // 256,
            epochs=15,
            verbose=2,
            callbacks=[
                EarlyStopping(
                    patience=5,
                    restore_best_weights=True
                )
            ],
            validation_data=FitGenerator(self.__valid_feature, self.__valid_label, False, 1, None),
            validation_steps=self.__valid_feature.shape[0]
        )

        self.__test_index["label"] = np.argmax(
            self.__vgg_net.predict_generator(
                generator=PredictGenerator(self.__test_feature),
                steps=self.__test_feature.shape[0]),
            axis=1)

    def data_write(self):
        self.__test_index.to_csv(os.path.join(self.__path, "sample_submission.csv"), index=False)


if __name__ == "__main__":
    vggp = VggNetPretrain(path="E:\\Kaggle\\Kannada_MNIST")
    vggp.data_read()
    vggp.data_prepare()
    vggp.model_fit_predict()
    vggp.data_write()
