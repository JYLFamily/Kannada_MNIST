# codeing:utf-8

import os
import gc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import Model
from keras.utils import Sequence
from keras.optimizers import SGD
from keras.initializers import he_normal
from keras.callbacks import LambdaCallback
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, ZeroPadding2D, Conv2D, BatchNormalization, Activation, Add
from keras.layers import GlobalAveragePooling2D, Flatten, Dense
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


def residual_net():
    # input layer
    input_layer = Input(shape=(28, 28, 1))
    x = ZeroPadding2D(padding=2, data_format="channels_last")(input_layer)
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


class StoppingCriteria(object):
    def __init__(self, smooth=0.3, min_iter=20):
        self.__smooth = smooth
        self.__num_iter, self.__min_iter = 0, min_iter
        self.__fir_loss, self.__smo_loss = [None for _ in range(2)]

    def __call__(self, loss):
        self.__num_iter += 1

        if self.__fir_loss is None:
            self.__fir_loss = loss

        if self.__smo_loss is None:
            self.__smo_loss = loss
        else:
            self.__smo_loss = ((1 - self.__smooth) * loss) + (self.__smooth * self.__smo_loss)

        print(
            "num iter {0}, smo loss {1}, fir loss {2}".format(self.__num_iter, self.__smo_loss, self.__fir_loss))
        return (self.__smo_loss > self.__fir_loss * 2) and (self.__num_iter >= self.__min_iter)


class LearningRateFinder(object):
    def __init__(self, *, net, lr_s, lr_f):
        self.__net = net
        self.__lr_s, self.__lr_f = lr_s, lr_f

        self.__lr_list, self.__loss_list = [[] for _ in range(2)]
        self.__stopping_criteria = StoppingCriteria()

    def find(self, trn_x, trn_y, image_augment):
        def on_batch_end(logs):
            lr = K.get_value(self.__net.optimizer.lr)
            self.__lr_list.append(lr)

            loss = logs["loss"]
            self.__loss_list.append(loss)

            if self.__stopping_criteria(loss):
                self.__net.stop_training = True

                return

            lr *= self.__lr_f
            K.set_value(self.__net.optimizer.lr, lr)

        self.__net.compile(optimizer=SGD(self.__lr_f), loss="sparse_categorical_crossentropy")
        self.__net.fit_generator(
            generator=FitGenerator(trn_x, trn_y, 128, image_augment),
            steps_per_epoch=trn_x.shape[0] // 128,
            epochs=99,
            verbose=0,
            callbacks=[LambdaCallback(on_batch_end=lambda _, logs: on_batch_end(logs))]
        )

    def plot(self):
        _, ax = plt.subplots()
        ax = sns.scatterplot(self.__lr_list, self.__loss_list)
        ax.set_xlabel("Learning Rate")
        ax.set_ylabel("Loss")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim([self.__lr_list[0], self.__lr_list[-1]])
        ax.set_ylim([min(self.__loss_list) * 0.8, self.__loss_list[0] * 4])
        plt.show()


if __name__ == "__main__":
    train = pd.read_csv(os.path.join("G:\\Kaggle\\Kannada_MNIST", "train.csv"))
    train = train.sample(frac=1, random_state=7).reset_index(drop=True)
    train_feature, train_label = (
        train.iloc[:, 1:].copy(deep=True), train.iloc[:, 0].copy(deep=True))
    del train
    gc.collect()

    train_feature, train_label = train_feature.to_numpy(), train_label.to_numpy()
    train_feature = train_feature.reshape((-1, 28, 28, 1))

    image_data_generator = ImageDataGenerator(
        rotation_range=15,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        data_format="channels_last"
    )

    lrf = LearningRateFinder(net=residual_net(), lr_s=1e-10, lr_f=1.1)
    lrf.find(trn_x=train_feature, trn_y=train_label, image_augment=image_data_generator)
    lrf.plot()


