# coding:utf-8

import os
import gc
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)


class LeNet(object):
    def __init__(self, *, path):
        self.__path = path
        self.__train, self.__valid, self.__test = [None for _ in range(3)]
        self.__train_feature, self.__valid_feature, self.__test_feature = [None for _ in range(3)]
        self.__train_label, self.__valid_label, self.__test_index = [None for _ in range(3)]

        self.__le_net = None

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

        self.__train_feature = self.__train_feature / 255
        self.__valid_feature = self.__valid_feature / 255
        self.__test_feature = self.__test_feature / 255

        self.__train_feature = self.__train_feature.reshape((-1, 28, 28, 1))
        self.__valid_feature = self.__valid_feature.reshape((-1, 28, 28, 1))
        self.__test_feature = self.__test_feature.reshape((-1, 28, 28, 1))

    def model_fit_predict(self):
        self.__le_net = Sequential([
            Conv2D(
                filters=6,
                kernel_size=(5, 5),
                data_format="channels_last",
                activation="sigmoid"
            ),
            MaxPool2D(pool_size=(2, 2), strides=2, data_format="channels_last"),
            Conv2D(
                filters=16,
                kernel_size=(5, 5),
                data_format="channels_last",
                activation="sigmoid"
            ),
            MaxPool2D(pool_size=(2, 2), strides=2, data_format="channels_last"),
            Flatten(),
            Dense(units=120, activation="sigmoid"),
            Dense(units=84, activation="sigmoid"),
            Dense(units=10, activation="softmax")
        ])

        self.__le_net.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        self.__le_net.fit(
            x=self.__train_feature,
            y=self.__train_label,
            batch_size=256,
            epochs=15,
            verbose=2,
            callbacks=[
                EarlyStopping(
                    patience=5,
                    restore_best_weights=True
                )
            ],
            validation_data=(self.__valid_feature, self.__valid_label)
        )
        self.__test_index["label"] = np.argmax(self.__le_net.predict(self.__test_feature), axis=1)

    def data_write(self):
        self.__test_index.to_csv(os.path.join(self.__path, "sample_submission.csv"), index=False)


if __name__ == "__main__":
    lt = LeNet(path="G:\\Kaggle\\Kannada_MNIST")
    lt.data_read()
    lt.data_prepare()
    lt.model_fit_predict()
    lt.data_write()


