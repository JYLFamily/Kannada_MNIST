# coding:utf-8

import os
import gc
import numpy as np
import pandas as pd
from keras.models import Model
from keras.utils import Sequence
from keras.initializers import he_normal
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, ZeroPadding2D, Conv2D, BatchNormalization, Activation, Add
from keras.layers import GlobalAveragePooling2D, Dense
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)


def residual(input_tensor, filters, strides, flag):
    x = ZeroPadding2D(padding=1, data_format="channels_last")(input_tensor)
    x = Conv2D(
        filters=filters,
        kernel_size=3,
        strides=strides,
        data_format="channels_last",
        kernel_initializer=he_normal(7))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = ZeroPadding2D(padding=1, data_format="channels_last")(x)
    x = Conv2D(
        filters=filters,
        kernel_size=3,
        data_format="channels_last",
        kernel_initializer=he_normal(7))(x)
    x = BatchNormalization()(x)

    if flag:
        input_tensor = Conv2D(
            filters=filters,
            kernel_size=1,
            strides=strides,
            data_format="channels_last",
            kernel_initializer=he_normal(7))(input_tensor)

    return Activation("relu")(Add()([x, input_tensor]))


class FitGenerator(Sequence):
    def __init__(self, feature, label, batch_size, image_augment):
        self.__index = np.arange(feature.shape[0])
        self.__feature, self.__label = feature, label
        self.__batch_size, self.__image_augment = batch_size, image_augment

    def __len__(self):
        return self.__feature.shape[0] // self.__batch_size

    def __getitem__(self, idx):
        index = self.__index[idx * self.__batch_size: (idx + 1) * self.__batch_size]
        batch_feature, batch_label = (
            np.array([image / 255 for image in self.__feature[index]]), self.__label[index])

        if self.__image_augment is not None:
            batch_feature, batch_label = (
                next(self.__image_augment.flow(np.array(batch_feature), batch_label, batch_size=self.__batch_size)))

        return batch_feature, batch_label

    def on_epoch_end(self):
        np.random.shuffle(self.__index)


class PredictGenerator(Sequence):
    def __init__(self, feature):
        self.__index = np.arange(feature.shape[0])
        self.__feature = feature

    def __len__(self):
        return self.__feature.shape[0]

    def __getitem__(self, idx):
        index = self.__index[idx: (idx + 1)]

        batch_feature = np.array([image / 255 for image in self.__feature[index]])

        return batch_feature


class ResNet(object):
    def __init__(self, *, path):
        self.__path = path
        self.__train, self.__valid, self.__test = [None for _ in range(3)]
        self.__train_feature, self.__valid_feature, self.__test_feature = [None for _ in range(3)]
        self.__train_label, self.__valid_label, self.__test_index = [None for _ in range(3)]

        self.__image_data_generator = None
        self.__res_net = None

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
        # input layer
        input_layer = Input(shape=(28, 28, 1))
        x = ZeroPadding2D(padding=1, data_format="channels_last")(input_layer)
        x = Conv2D(
            filters=64,
            kernel_size=3,
            data_format="channels_last",
            kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        # residual block 1
        x = residual(input_tensor=x, filters=64, strides=1, flag=False)
        x = residual(input_tensor=x, filters=64, strides=1, flag=False)

        # residual block 2
        x = residual(input_tensor=x, filters=128, strides=2, flag=True)
        x = residual(input_tensor=x, filters=128, strides=1, flag=False)

        # residual block 3
        x = residual(input_tensor=x, filters=256, strides=2, flag=True)
        x = residual(input_tensor=x, filters=256, strides=1, flag=False)

        # output layer
        x = GlobalAveragePooling2D()(x)
        output_layer = Dense(units=10, activation="softmax", kernel_initializer=he_normal(7))(x)

        self.__res_net = Model(inputs=input_layer, outputs=output_layer)
        self.__res_net.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        self.__res_net.fit_generator(
            generator=FitGenerator(self.__train_feature, self.__train_label, 256, self.__image_data_generator),
            steps_per_epoch=self.__train_feature.shape[0] // 256,
            epochs=15,
            verbose=1,
            callbacks=[
                EarlyStopping(
                    patience=5,
                    restore_best_weights=True
                )
            ],
            validation_data=FitGenerator(self.__valid_feature, self.__valid_label, 512, None),
            validation_steps=self.__valid_feature.shape[0],
            workers=1,
            use_multiprocessing=False
        )

        self.__test_index["label"] = np.argmax(
            self.__res_net.predict_generator(
                generator=PredictGenerator(self.__test_feature),
                steps=self.__test_feature.shape[0],
                workers=1,
                use_multiprocessing=False),
            axis=1)

    def data_write(self):
        self.__test_index.to_csv(os.path.join(self.__path, "sample_submission.csv"), index=False)


if __name__ == "__main__":
    rn = ResNet(path="G:\\Kaggle\\Kannada_MNIST")
    rn.data_read()
    rn.data_prepare()
    rn.model_fit_predict()
    rn.data_write()
