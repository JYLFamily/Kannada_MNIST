# coding:utf-8

import os
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
np.random.seed(7)
tf.random.set_seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)


class ResNet(object):
    def __init__(self, *, path):
        self.__path = path
        self.__train, self.__test = [None for _ in range(2)]
        self.__train_feature, self.__test_feature = [None for _ in range(2)]
        self.__train_label, self.__test_index = [None for _ in range(2)]

        self.__image_data_generator = None

    def data_read(self):
        self.__train = pd.read_csv(os.path.join(self.__path, "train.csv"))
        self.__test = pd.read_csv(os.path.join(self.__path, "test.csv"))

    def data_prepare(self):
        self.__train_feature, self.__train_label = (
            self.__train.iloc[:, 1:].copy(deep=True), self.__train.iloc[:, 0].copy(deep=True))
        self.__test_feature, self.__test_index = (
            self.__test.iloc[:, 1:].copy(deep=True), self.__test.iloc[:, [0]].copy(deep=True))
        del self.__train, self.__test
        gc.collect()

        self.__train_feature, self.__train_label = self.__train_feature.to_numpy(), self.__train_label.to_numpy()
        self.__test_feature = self.__test_feature.to_numpy()

        self.__train_feature = self.__train_feature.reshape((-1, 28, 28, 1))
        self.__test_feature = self.__test_feature.reshape((-1, 28, 28, 1))

        self.__image_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=15,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
            data_format="channels_last"
        )


if __name__ == "__main__":
    pass