# coding:utf-8

import os
import gc
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)


def fit_generator(feature, label, batch_size, image_augment):
    index = np.arange(len(feature))
    batch = []

    while True:
        np.random.shuffle(index)

        for idx in index:
            batch.append(idx)
            if len(batch) == batch_size:
                batch_feature, batch_label = (
                    np.array([zoom(image, zoom=(8, 8, 1)) / 255 for image in feature[batch]]), label[batch])
                if image_augment is not None:
                    batch_feature, batch_label = (
                        next(image_augment.flow(np.array(batch_feature), batch_label, batch_size=batch_size)))

                yield batch_feature, batch_label

                batch.clear()


def predict_generator(feature, batch_size=1):
    index = np.arange(len(feature))
    batch = []

    while True:
        for idx in index:
            batch.append(idx)
            if len(batch) == batch_size:
                yield np.array([zoom(image, zoom=(8, 8, 1)) / 255 for image in feature[batch]])

                batch.clear()


class AlexNet(object):
    def __init__(self, *, path):
        self.__path = path
        self.__train, self.__valid, self.__test = [None for _ in range(3)]
        self.__train_feature, self.__valid_feature, self.__test_feature = [None for _ in range(3)]
        self.__train_label, self.__valid_label, self.__test_index = [None for _ in range(3)]

        self.__image_data_generator = None
        self.__alex_net = None

    def data_read(self):
        self.__train = pd.read_csv(os.path.join(self.__path, "train.csv")).head(1028)
        self.__valid = pd.read_csv(os.path.join(self.__path, "Dig-MNIST.csv")).head(1028)
        self.__test = pd.read_csv(os.path.join(self.__path, "test.csv")).head(1028)

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
        self.__alex_net = Sequential([
            Conv2D(
                filters=96,
                kernel_size=(11, 11),
                strides=4,
                data_format="channels_last",
                activation="relu",
                input_shape=(224, 224, 1)
            ),
            MaxPool2D(pool_size=(3, 3), strides=2, data_format="channels_last"),
            Conv2D(
                filters=256,
                kernel_size=(5, 5),
                padding="same",
                data_format="channels_last",
                activation="relu"
            ),
            MaxPool2D(pool_size=(3, 3), strides=2, data_format="channels_last"),
            Conv2D(
                filters=384,
                kernel_size=(3, 3),
                padding="same",
                data_format="channels_last",
                activation="relu"
            ),
            Conv2D(
                filters=384,
                kernel_size=(3, 3),
                padding="same",
                data_format="channels_last",
                activation="relu"
            ),
            Conv2D(
                filters=256,
                kernel_size=(3, 3),
                padding="same",
                data_format="channels_last",
                activation="relu"
            ),
            MaxPool2D(pool_size=(3, 3), strides=2, data_format="channels_last",),
            Flatten(),
            Dense(units=4096, activation="relu"),
            Dropout(0.5),
            Dense(units=4096, activation="relu"),
            Dropout(0.5),
            Dense(units=10, activation="softmax")
        ])

        self.__alex_net.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        self.__alex_net.fit_generator(
            generator=fit_generator(self.__train_feature, self.__train_label, 256, self.__image_data_generator),
            steps_per_epoch=self.__train_feature.shape[0] // 256,
            epochs=2,
            verbose=1,
            validation_data=fit_generator(self.__valid_feature, self.__valid_label, 256, None),
            validation_steps=self.__valid_feature.shape[0] // 256
        )

        self.__test_index["label"] = np.argmax(
            self.__alex_net.predict_generator(
                generator=predict_generator(self.__test_feature),
                steps=self.__test_feature.shape[0]),
            axis=1)

    def data_write(self):
        self.__test_index.to_csv(os.path.join(self.__path, "sample_submission.csv"), index=False)


if __name__ == "__main__":
    at = AlexNet(path="G:\\Kaggle\\Kannada_MNIST")
    at.data_read()
    at.data_prepare()
    at.model_fit_predict()
    at.data_write()