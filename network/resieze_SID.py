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

#block 8
up8 = UpSampleConcat(64, 128,name = "Variable_2")(conv3,conv2)
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
    return keras.Model(inputs=[input], outputs= output), "SID5"

# [[[0.08602792 0.07336402 0.03864708]
#   [0.07903548 0.06857046 0.03974826]
#   [0.07302186 0.06523632 0.03991566]
#   ...
#   [0.08560883 0.0706325  0.03421818]
#   [0.08870581 0.07592083 0.04423552]
#   [0.09269468 0.07398462 0.04510165]]
#
#  [[0.08639354 0.06926662 0.04080523]
#   [0.07495556 0.05871126 0.03806867]
#   [0.0648265  0.05128469 0.0295438 ]
#   ...
#   [0.07457116 0.05376681 0.02442185]
#   [0.08547896 0.06378847 0.03566743]
#   [0.09074153 0.07285322 0.04212515]]
#
#  [[0.08094521 0.06693228 0.03533838]
#   [0.06976546 0.05532891 0.03149667]
#   [0.06147292 0.03819069 0.0226363 ]
#   ...
#   [0.05881179 0.04735892 0.0131845 ]
#   [0.07693516 0.05631597 0.03127469]
#   [0.0918835  0.07338641 0.03935628]]
#
#  ...
#
#  [[0.14324224 0.1130197  0.07185313]
#   [0.13929786 0.11061344 0.07296891]
#   [0.15318054 0.11301191 0.06830302]
#   ...
#   [0.1909389  0.1492533  0.1050798 ]
#   [0.17032841 0.13691138 0.09747595]
#   [0.1539396  0.12073004 0.0849484 ]]
#
#  [[0.1305592  0.10794487 0.06947236]
#   [0.12911822 0.10874669 0.07035416]
#   [0.13768901 0.10447089 0.0692339 ]
#   ...
#   [0.16724506 0.13411611 0.09991733]
#   [0.15323672 0.12826121 0.09729282]
#   [0.13825905 0.11183639 0.07992917]]
#
#  [[0.12081882 0.10244236 0.06949592]
#   [0.11788579 0.09819439 0.06849099]
#   [0.13223128 0.10482494 0.06660041]
#   ...
#   [0.1479801  0.11805215 0.08593237]
#   [0.1445211  0.11821326 0.08449197]
#   [0.12433289 0.10191898 0.07100058]]]