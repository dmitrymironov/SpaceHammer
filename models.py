import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class FlowNet(keras.Model):
    def __init__(self):
        super(FlowNet, self).__init__()

        self.conv1 = layers.Conv2D(64, kernel_size=7, strides=2, padding='same', input_shape=(height, width, 6),
                              name='conv1')
        self.leaky_relu_1 = layers.LeakyReLU(0.1)

        self.conv2 = layers.Conv2D(128, kernel_size=5, strides=2,
                              padding='same', name='conv2')
        self.leaky_relu_2 = layers.LeakyReLU(0.1)

        self.conv3 = layers.Conv2D(256, kernel_size=5, strides=2,
                              padding='same', name='conv3')
        self.leaky_relu_3 = layers.LeakyReLU(0.1)

        self.conv3_1 = layers.Conv2D(
            256, kernel_size=3, strides=1, padding='same', name='conv3_1')
        self.leaky_relu_3_1 = layers.LeakyReLU(0.1)

        self.conv4 = layers.Conv2D(512, kernel_size=3, strides=2,
                              padding='same', name='conv4')
        self.leaky_relu_4 = layers.LeakyReLU(0.1)

        self.conv4_1 = layers.Conv2D(
            512, kernel_size=3, strides=1, padding='same', name='conv4_1')
        self.leaky_relu_4_1 = layers.LeakyReLU(0.1)

        self.conv5 = layers.Conv2D(512, kernel_size=3, strides=2,
                              padding='same', name='conv5')
        self.leaky_relu_5 = layers.LeakyReLU(0.1)

        self.conv5_1 = layers.Conv2D(
            512, kernel_size=3, strides=1, padding='same', name='conv5_1')
        self.leaky_relu_5_1 = layers.LeakyReLU(0.1)

        self.conv6 = layers.Conv2D(1024, kernel_size=3, strides=2,
                              padding='same', name='conv6')
        self.leaky_relu_6 = layers.LeakyReLU(0.1)

    def call(self, inputs, is_training=False):
        x = self.conv1(inputs)
        x = self.leaky_relu_1(x)
        x = self.conv2(x)
        x = self.leaky_relu_2(x)
        x = self.conv3(x)
        x = self.leaky_relu_3(x)
        x = self.conv3_1(x)
        x = self.leaky_relu_3_1(x)
        x = self.conv4(x)
        x = self.leaky_relu_4(x)
        x = self.conv4_1(x)
        x = self.leaky_relu_4_1(x)
        x = self.conv5(x)
        x = self.leaky_relu_5(x)
        x = self.conv5_1(x)
        x = self.leaky_relu_5_1(x)
        x = self.conv6(x)
        x = self.leaky_relu_6(x)
        return x

class PoseConvGRUNet(keras.Model):
    def __init__(self):
        super(PoseConvGRUNet, self).__init__()
        self.max_pooling = layers.MaxPool2D(2, strides=2)
        self.reshape = layers.Reshape((-1, 5 * 1 * 1024))
        self.gru = layers.GRU(3)
        self.dense_1 = layers.Dense(4096)
        self.leaky_relu_1 = layers.LeakyReLU(0.1)
        self.dense_2 = layers.Dense(1024)
        self.leaky_relu_2 = layers.LeakyReLU(0.1)
        self.dense_3 = layers.Dense(128)
        self.leaky_relu_3 = layers.LeakyReLU(0.1)
        self.out = layers.Dense(6)

    def call(self, inputs, is_training=False):
        x = self.max_pooling(inputs)
        x = self.reshape(x)
        x = self.gru(x)
        x = self.dense_1(x)
        x = self.leaky_relu_1(x)
        x = self.dense_2(x)
        x = self.leaky_relu_2(x)
        x = self.dense_3(x)
        x = self.leaky_relu_3(x)
        x = self.out(x)
        return x
