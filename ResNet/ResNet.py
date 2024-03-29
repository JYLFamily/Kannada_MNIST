# coding:utf-8

import os
import gc
import numpy as np
import pandas as pd
from keras.layers import *
from keras.models import Model
from keras.utils import Sequence
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from keras.initializers import he_normal
from sklearn.model_selection import KFold
from scikitplot.metrics import plot_confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
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


def residual_net(output="binary"):
    # input layer
    input_layer = Input(shape=(28, 28, 1))
    x = ZeroPadding2D(padding=1, data_format="channels_last")(input_layer)
    x = Conv2D(
        filters=64,
        kernel_size=3,
        data_format="channels_last",
        kernel_initializer=he_normal(7))(x)
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
    if output == "binary":
        x = GlobalAveragePooling2D()(x)
        output_layer = Dense(units=1, activation="sigmoid", kernel_initializer=he_normal(7))(x)
    else:
        x = GlobalAveragePooling2D()(x)
        output_layer = Dense(units=10, activation="softmax", kernel_initializer=he_normal(7))(x)

    return Model(inputs=input_layer, outputs=output_layer)


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
        self.__train, self.__test = [None for _ in range(2)]
        self.__train_feature, self.__test_feature = [None for _ in range(2)]
        self.__train_label, self.__test_index = [None for _ in range(2)]

        self.__folds = None
        self.__sub_preds = None
        self.__sub_mixed = None
        self.__val_preds = None
        self.__val_mixed = None

        self.__image_data_generator = None
        self.__res_net = None

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

        self.__image_data_generator = ImageDataGenerator(
            rotation_range=15,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
            data_format="channels_last"
        )

    def model_fit_predict(self):
        self.__folds = KFold(n_splits=5, shuffle=True, random_state=7)
        self.__sub_preds = np.zeros(shape=(self.__test_feature.shape[0],  10))
        self.__val_preds = np.zeros(shape=(self.__train_feature.shape[0], 10))

        # 1. network
        for n_fold, (trn_idx, val_idx) in enumerate(self.__folds.split(
                    X=self.__train_feature, y=self.__train_label)):
            print("Fold: " + str(n_fold))
            trn_x = np.copy(self.__train_feature[trn_idx])
            val_x = np.copy(self.__train_feature[val_idx])
            tes_x = np.copy(self.__test_feature)

            trn_y = np.copy(self.__train_label[trn_idx])
            val_y = np.copy(self.__train_label[val_idx])

            self.__res_net = residual_net(output="softmax")
            self.__res_net.compile(optimizer=Adam(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            self.__res_net.fit_generator(
                generator=FitGenerator(trn_x, trn_y, 256, self.__image_data_generator),
                steps_per_epoch=trn_x.shape[0] // 256,
                epochs=60,
                verbose=1,
                callbacks=[
                    ReduceLROnPlateau(
                        patience=3
                    ),
                    EarlyStopping(
                        patience=6,
                        restore_best_weights=True
                    )
                ],
                validation_data=FitGenerator(val_x, val_y, 256, None),
                validation_steps=val_x.shape[0] // 256,
                workers=1,
                use_multiprocessing=False
            )

            self.__sub_preds += self.__res_net.predict_generator(
                generator=PredictGenerator(tes_x),
                steps=tes_x.shape[0],
                workers=1,
                use_multiprocessing=False) / self.__folds.n_splits

            self.__val_preds[val_idx, :] = self.__res_net.predict_generator(
                generator=PredictGenerator(val_x),
                steps=val_x.shape[0],
                workers=1,
                use_multiprocessing=False)

        # 2. network
        tra_index = np.where(np.logical_or(
            self.__train_label == 0,
            self.__train_label == 1
        ))[0].tolist()
        tes_index = np.where(np.logical_or(
            np.argmax(self.__sub_preds, axis=1) == 0,
            np.argmax(self.__sub_preds, axis=1) == 1
        ))[0].tolist()

        self.__folds = KFold(n_splits=5, shuffle=True, random_state=7)
        self.__sub_mixed = np.zeros(shape=(self.__test_feature[tes_index].shape[0], ))
        self.__val_mixed = np.zeros(shape=(self.__train_feature[tra_index].shape[0],))

        for n_fold, (trn_idx, val_idx) in enumerate(self.__folds.split(
                    X=self.__train_feature[tra_index], y=self.__train_label[tra_index])):
            print("Fold: " + str(n_fold))
            trn_x = np.copy(self.__train_feature[tra_index][trn_idx])
            val_x = np.copy(self.__train_feature[tra_index][val_idx])
            tes_x = np.copy(self.__test_feature[tes_index])

            trn_y = np.copy(self.__train_label[tra_index][trn_idx])
            val_y = np.copy(self.__train_label[tra_index][val_idx])

            self.__res_net = residual_net(output="binary")
            self.__res_net.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
            self.__res_net.fit_generator(
                generator=FitGenerator(trn_x, trn_y, 256, self.__image_data_generator),
                steps_per_epoch=trn_x.shape[0] // 256,
                epochs=60,
                verbose=1,
                callbacks=[
                    ReduceLROnPlateau(
                        patience=3
                    ),
                    EarlyStopping(
                        patience=6,
                        restore_best_weights=True
                    )
                ],
                validation_data=FitGenerator(val_x, val_y, 256, None),
                validation_steps=val_x.shape[0] // 256,
                workers=1,
                use_multiprocessing=False
            )

            self.__sub_mixed += self.__res_net.predict_generator(
                generator=PredictGenerator(tes_x),
                steps=tes_x.shape[0],
                workers=1,
                use_multiprocessing=False).reshape(-1, ) / self.__folds.n_splits

            self.__val_mixed[val_idx] = self.__res_net.predict_generator(
                generator=PredictGenerator(val_x),
                steps=val_x.shape[0],
                workers=1,
                use_multiprocessing=False).reshape(-1, )

        self.__sub_preds = np.argmax(self.__sub_preds, axis=1)
        self.__sub_preds[tes_index] = np.where(self.__sub_mixed > 0.5, 1, 0)

        self.__val_preds = np.argmax(self.__val_preds, axis=1)
        self.__val_preds[tra_index] = np.where(self.__val_mixed > 0.5, 1, 0)

    def data_write(self):
        self.__test_index["label"] = self.__sub_preds
        self.__test_index.to_csv("submission.csv", index=False)

    def plot_image(self):
        plot_confusion_matrix(self.__train_label, self.__val_preds, normalize=True)
        plt.show()


if __name__ == "__main__":
    rn = ResNet(path="D:\\Kaggle\\Kannada_MNIST")
    rn.data_read()
    rn.data_prepare()
    rn.model_fit_predict()
    rn.data_write()

