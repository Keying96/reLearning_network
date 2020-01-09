#!/usr/bin/env python
# encoding: utf-8
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from network.UpSampleConcat import UpSampleConcat as UpSampleConcat

# 自定义激活函数
def lrelu(x):
    return tf.maximum(x * 0.2, x)

#define input layer
input = layers.Input(shape=[1424, 2128, 4])

#block 1
conv1 = layers.Conv2D(32, [3, 3], padding='SAME', activation=lrelu, name='g_conv1_1')(input)
conv1 = layers.Conv2D(32, [3, 3], padding='SAME', activation=lrelu, name='g_conv1_2')(conv1)
pool1 = layers.MaxPool2D(pool_size=(2, 2), padding='SAME')(conv1)

#block 2
conv2 = layers.Conv2D(64, [3, 3], padding='SAME', activation=lrelu, name='g_conv2_1')(pool1)
conv2 = layers.Conv2D(64, [3, 3], padding='SAME', activation=lrelu, name='g_conv2_2')(conv2)
pool2 = layers.MaxPool2D([2, 2], padding='SAME')(conv2)

#block 3
conv3 = layers.Conv2D(128, [3, 3], padding='SAME', activation=lrelu, name='g_conv3_1')(pool2)
conv3 = layers.Conv2D(128, [3, 3], padding='SAME', activation=lrelu, name='g_conv3_2')(conv3)
pool3 = layers.MaxPool2D([2, 2], padding='SAME')(conv3)

#block 4
conv4 = layers.Conv2D(256, [3, 3], padding='SAME', activation=lrelu, name='g_conv4_1')(pool3)
conv4 = layers.Conv2D(256, [3, 3], padding='SAME', activation=lrelu, name='g_conv4_2')(conv4)
pool4 = layers.MaxPool2D([2, 2], padding='SAME')(conv4)

#block 5
conv5 = layers.Conv2D(512, [3, 3], padding='SAME', activation=lrelu, name='g_conv5_1')(pool4)
conv5 = layers.Conv2D(512, [3, 3], padding='SAME', activation=lrelu, name='g_conv5_2')(conv5)

#block 6
up6 = UpSampleConcat(256, 512, name = "Variable")(conv5,conv4)
# up6 = layers.Conv2DTranspose(256, (2,2), (1,1), 'same',name = "Variable")(conv5)
# up6 = layers.concatenate([conv5,conv4], axis = 3)
conv6 = layers.Conv2D(256, [3, 3], padding='SAME', activation=lrelu, name='g_conv6_1')(up6)
conv6 = layers.Conv2D(256, [3, 3], padding='SAME', activation=lrelu, name='g_conv6_2')(conv6)

#block 7
up7 = UpSampleConcat(128, 256,name = "Variable_1")(conv6,conv3)
conv7 = layers.Conv2D(128, [3, 3], padding='SAME', activation=lrelu, name='g_conv7_1')(up7)
conv7 = layers.Conv2D(128, [3, 3], padding='SAME', activation=lrelu, name='g_conv7_2')(conv7)

#block 8
up8 = UpSampleConcat(64, 128,name = "Variable_2")(conv7,conv2)
conv8 = layers.Conv2D(64, [3, 3], padding='SAME', activation=lrelu, name='g_conv8_1')(up8)
conv8 = layers.Conv2D(64, [3, 3], padding='SAME', activation=lrelu, name='g_conv8_2')(conv8)

#block 9
up9 = UpSampleConcat(32, 64,name = "Variable_3")(conv8,conv1)
conv9 = layers.Conv2D(32, [3, 3], padding='SAME', activation=lrelu, name='g_conv9_1')(up9)
conv9 = layers.Conv2D(32, [3, 3], padding='SAME', activation=lrelu, name='g_conv9_2')(conv9)

#block 10
conv10 = layers.Conv2D(12, [1, 1], padding='SAME', activation=None, name='g_conv10')(conv9)
output = tf.nn.depth_to_space(conv10, 2)

#build model
def SID():
    # return keras.Model(inputs=[input], outputs= [output,pool1, up6,conv6,conv5,conv4])
    return keras.Model(inputs=[input], outputs= output)


