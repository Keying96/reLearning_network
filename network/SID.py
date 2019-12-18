#!/usr/bin/env python
# encoding: utf-8
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 自定义上采样函数
def upsample_and_concat(x1, x2, output_channels, in_channels):
    pool_size = 2
    deconv_filter = tf.Variable(tf.random.truncated_normal([pool_size, pool_size, output_channels, in_channels],stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])
    deconv_output = layers.concatenate([deconv,x2],3)
    deconv_output.set_shape([None, None, None, output_channels * 2])

    return deconv_output

# 自定义激活函数
def lrelu(x):
    return tf.maximum(x * 0.2, x)

input = layers.Input(shape=[256, 256, 1])
conv1 = layers.Conv2D(32, 3, padding='SAME', activation=lrelu, name='g_conv1_1')(input)
conv1 = layers.Conv2D(32, 3, padding='SAME', activation=lrelu, name='g_conv1_2')(conv1)
pool1 = layers.MaxPool2D(pool_size=(2, 2), padding='SAME')(conv1)

conv2 = layers.Conv2D(64, [3, 3], padding='SAME', activation=lrelu, name='g_conv2_1')(pool1)
conv2 = layers.Conv2D(64, [3, 3], padding='SAME', activation=lrelu, name='g_conv2_2')(conv2)
pool2 = layers.MaxPool2D([2, 2], padding='SAME')(conv2)

conv3 = layers.Conv2D(128, [3, 3], padding='SAME', activation=lrelu, name='g_conv3_1')(pool2)
conv3 = layers.Conv2D(128, [3, 3], padding='SAME', activation=lrelu, name='g_conv3_2')(conv3)
pool3 = layers.MaxPool2D([2, 2], padding='SAME')(conv3)

conv4 = layers.Conv2D(256, [3, 3], padding='SAME', activation=lrelu, name='g_conv4_1')(pool3)
conv4 = layers.Conv2D(256, [3, 3], padding='SAME', activation=lrelu, name='g_conv4_2')(conv4)
pool4 = layers.MaxPool2D([2, 2], padding='SAME')(conv4)

conv5 = layers.Conv2D(512, [3, 3], padding='SAME', activation=lrelu, name='g_conv5_1')(pool4)
conv5 = layers.Conv2D(512, [3, 3], padding='SAME', activation=lrelu, name='g_conv5_2')(conv5)

up6 = upsample_and_concat(conv5, conv4, 256, 512)
conv6 = layers.Conv2D(256, [3, 3], padding='SAME', activation=lrelu, name='g_conv6_1')(up6)
conv6 = layers.Conv2D(256, [3, 3], padding='SAME', activation=lrelu, name='g_conv6_2')(conv6)
up7 = upsample_and_concat(conv6, conv3, 128, 256)
conv7 = layers.Conv2D(128, [3, 3], padding='SAME', activation=lrelu, name='g_conv7_1')(up7)
conv7 = layers.Conv2D(128, [3, 3], padding='SAME', activation=lrelu, name='g_conv7_2')(conv7)

up8 = upsample_and_concat(conv7, conv2, 64, 128)
conv8 = layers.Conv2D(64, [3, 3], padding='SAME', activation=lrelu, name='g_conv8_1')(up8)
conv8 = layers.Conv2D(64, [3, 3], padding='SAME', activation=lrelu, name='g_conv8_2')(conv8)

up9 = upsample_and_concat(conv8, conv1, 32, 64)
conv9 = layers.Conv2D(32, [3, 3], padding='SAME', activation=lrelu, name='g_conv9_1')(up9)
conv9 = layers.Conv2D(32, [3, 3], padding='SAME', activation=lrelu, name='g_conv9_2')(conv9)

conv10 = layers.Conv2D(12, [1, 1], padding='SAME', activation=None, name='g_conv10')(conv9)
output = tf.nn.depth_to_space(conv10, 2)
# return  output

# model = keras.Model(inputs=input, outputs=output)

def SID():
    return keras.Model(inputs=input, outputs=output)
