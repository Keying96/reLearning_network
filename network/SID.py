#!/usr/bin/env python
# encoding: utf-8
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from network.UpSampleConcat import UpSampleConcat as UpSampleConcat
from network import tool

output_dir = "./reduce_layer/concat/"
model_name = 'SID'
# 自定义激活函数
def lrelu(x):
    return tf.maximum(x * 0.2, x)

#define input layer
# input = layers.Input(shape=[1424, 2128, 4])
input = layers.Input(shape=[1080,1920,4])


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
    return keras.Model(inputs=[input], outputs= [output, conv1]), model_name


# [[[0.05170439 0.04130556 0.02457193]
#   [0.04224291 0.04398747 0.01808512]
#   [0.03502263 0.03482097 0.02167565]
#   ...
#   [0.03027131 0.02045538 0.00977623]
#   [0.0436894  0.03676117 0.02933072]
#   [0.04960163 0.04767065 0.02955819]]
#
#  [[0.04433307 0.04613304 0.01944414]
#   [0.03668036 0.03158956 0.01338341]
#   [0.02390988 0.03853033 0.02861832]
#   ...
#   [0.01641053 0.01573719 0.01115987]
#   [0.02971533 0.01478659 0.01235973]
#   [0.04289033 0.03173914 0.02580509]]
#
#  [[0.03233987 0.02192079 0.01070535]
#   [0.02021698 0.01654663 0.00200317]
#   [0.0149487  0.01415355 0.00226102]
#   ...
#   [0.01595008 0.00931536 0.00902168]
#   [0.01594021 0.01874873 0.01243978]
#   [0.03380945 0.02765612 0.02164668]]
#
#  ...
#
#  [[0.17841052 0.13733554 0.09844085]
#   [0.1553575  0.11781669 0.08757261]
#   [0.14404035 0.09236483 0.06319447]
#   ...
#   [0.20256026 0.15953605 0.13969915]
#   [0.19027425 0.15405886 0.13181533]
#   [0.18481822 0.15002836 0.12283324]]
#
#  [[0.16667512 0.13090461 0.09535714]
#   [0.14625235 0.11948977 0.08806553]
#   [0.14095426 0.09809434 0.07250495]
#   ...
#   [0.19367583 0.15786423 0.13696972]
#   [0.17925838 0.15548451 0.13355258]
#   [0.17190577 0.14136635 0.11320847]]
#
#  [[0.1560206  0.12754107 0.09507387]
#   [0.14241089 0.1159236  0.09048896]
#   [0.14475685 0.11395253 0.08114915]
#   ...
#   [0.18900254 0.15651557 0.13071345]
#   [0.18220535 0.15526013 0.12403826]
#   [0.16399236 0.14208652 0.10823467]]]