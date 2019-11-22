# encoding: utf-8

import numpy as np
from scipy.special import expit
from keras.datasets import mnist
from keras.utils import to_categorical
np.random.seed(7)

inputs_units = 784
hidden_units = 256
output_units = 10


class MlpNumpy(object):
    def __init__(self):
        self.__hidden_weight = np.random.randn(hidden_units, inputs_units)
        self.__output_weight = np.random.randn(output_units, hidden_units)

    def forward(self, images_x):
        hidden_z = np.dot(self.__hidden_weight, images_x)
        hidden_a = expit(hidden_z)
        output_z = np.dot(self.__output_weight, hidden_a)
        output_a = expit(output_z)
        print(output_a)

        return hidden_z, hidden_a, output_z, output_a

    def backward(self, images_x, labels_y):
        """
        :param images_x: (inputs_units, batch_size)
        :param labels_y:
        :return:
        """
        # hidden_z, hidden_a (hidden_units, batch_size)
        # output_z, output_a (output_units, batch_size)
        hidden_z, hidden_a, output_z, output_a = self.forward(images_x)

        print("loss: {:.5f}".format(np.square(labels_y - output_a).mean().sum()))

        # delta_output (output_units, batch_size)
        delta_output = - (labels_y - output_a) * expit(output_z) * (1 - expit(output_z))
        # gradi_output_weight (output_units, hidden_units)
        gradi_output_weight = np.dot(delta_output, hidden_a.T)

        # hidden layer
        # delta_hidden (hidden_units, batch_size)
        delta_hidden = expit(hidden_z) * (1 - expit(hidden_z)) * np.dot(self.__output_weight.T, delta_output)
        # gradi_output_weight (hidden_units, inputs_units)
        gradi_hidden_weight = np.dot(delta_hidden, images_x.T)

        return gradi_output_weight, gradi_hidden_weight

    def train(self, epochs):
        (images_x, labels_y), (_, _) = mnist.load_data()

        images_x = images_x.reshape(784, -1)
        images_x = images_x.astype(np.float32)
        images_x = images_x / 255

        labels_y = to_categorical(labels_y, 10)
        labels_y = labels_y.reshape(10, -1)

        images_x = images_x[:, :256]
        labels_y = labels_y[:, :256]

        for _ in range(epochs):
            gradi_output_weight, gradi_hidden_weight = self.backward(images_x, labels_y)

            self.__output_weight -= 1e-2 * gradi_output_weight
            self.__hidden_weight -= 1e-2 * gradi_hidden_weight


if __name__ == "__main__":
    mn = MlpNumpy()
    mn.train(epochs=1000)



