import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# https://github.com/keras-team/keras/issues/5687

class FlowNet(keras.Model):
    def __init__(self, input_shape=(480, 640, 6)):
        super().__init__()

        self.conv1 = layers.Conv2D(64, kernel_size=7, strides=2, padding='same', input_shape=input_shape,
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
        self.leaky_relu_6 = layers.LeakyReLU(alpha=0.1)
        self.max_pooling = layers.MaxPool2D(2, strides=2)

    def compute_output_shape(self, input_shape):
        return (None,4,5,1024)

    def call(self, inputs):
        #assert inputs.shape==(480,640,6), "Incorrect FlowNet input shape"
        #x = layers.Reshape((480, 640, 6))(inputs)
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
        x = self.max_pooling(x)
        #self.max_pooling = layers.MaxPool2D(2, strides=2)
        return x

class PoseConvGRUNet(keras.Model):
    def __init__(self):
        super().__init__()
        #self.max_pooling = layers.MaxPool2D(2, strides=2)
        #self.reshape = layers.Reshape((-1, 5 * 1 * 1024))
        self.reshape = layers.Reshape((-1, 5 * 1 * 1024))
        self.gru = layers.GRU(3)
        self.dense_1 = layers.Dense(4096)
        self.leaky_relu_1 = layers.LeakyReLU(0.1)
        self.dense_2 = layers.Dense(1024)
        self.leaky_relu_2 = layers.LeakyReLU(0.1)
        self.dense_3 = layers.Dense(128)
        self.leaky_relu_3 = layers.LeakyReLU(0.1)
        #self.out = layers.Dense(6)
        self.out = layers.Dense(1) # only generating speed

    def call(self, x):
        #x = self.max_pooling(inputs)
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
'''
[batch size, temporal dimension, Ximg,Yimg,Ch]
'''
'''
class TopModel(keras.Model):
    def __init__(self, shape=(60, 480, 640, 6)):
        super().__init__()
        self.fn = layers.TimeDistributed(FlowNet())
        self.pcg = PoseConvGRUNet()        

    def call(self, x):
        # Our model gets sequence of frames and produces speed array
        assert inputs.shape == (
            60, 480, 640, 6), "Incorrect TopModel input shape"
        x = self.fn(x)
        x = self.pcg(x)
        return x
'''
