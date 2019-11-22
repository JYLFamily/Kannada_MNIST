# encoding: utf-8

import torch
import torch.nn.functional as F
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
np.random.seed(7)

inputs_units = 784
hidden_units = 256
output_units = 10


class MlpTorch(object):
    def __init__(self):
        self.__hidden_weight = torch.randn(hidden_units, inputs_units, requires_grad=True)
        self.__output_weight = torch.randn(output_units, hidden_units, requires_grad=True)

    def forward(self, images_x):
        hidden_z = torch.mm(self.__hidden_weight, images_x)
        hidden_a = torch.sigmoid(hidden_z)
        output_z = torch.mm(self.__output_weight, hidden_a)
        output_a = torch.sigmoid(output_z)

        return hidden_z, hidden_a, output_z, output_a

    def backward(self, images_x, labels_y):
        """
        :param images_x: (inputs_units, batch_size)
        :param labels_y:
        :return:
        """
        hidden_z, hidden_a, output_z, output_a = self.forward(images_x)
        loss = torch.pow(labels_y - output_a, 2).mean().sum()

        print("loss: {:.5f}".format(loss.item()))

        loss.backward()

        self.__output_weight.data -= 1e-2 * self.__output_weight.grad
        self.__hidden_weight.data -= 1e-2 * self.__hidden_weight.grad

        self.__output_weight.grad.zero_()
        self.__hidden_weight.grad.zero_()

    def train(self, epochs):
        (images_x, labels_y), (_, _) = mnist.load_data()

        images_x = images_x.reshape(784, 60000)
        images_x = images_x.astype(np.float32)
        images_x = images_x / 255

        labels_y = to_categorical(labels_y, 10)
        labels_y = labels_y.reshape(10, -1)

        images_x = images_x[:, :256]
        labels_y = labels_y[:, :256]

        images_x = torch.from_numpy(images_x)
        labels_y = torch.from_numpy(labels_y)

        for _ in range(epochs):
            self.backward(images_x, labels_y)


if __name__ == "__main__":
    mn = MlpTorch()
    mn.train(epochs=1000)