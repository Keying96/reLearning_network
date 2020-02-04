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

#block 9
up9 = UpSampleConcat(32, 64,name = "Variable_3")(conv2,conv1)
conv9 = layers.Conv2D(32, [3, 3], padding='SAME', activation=lrelu, name='g_conv9_1')(up9)
conv9 = layers.Conv2D(32, [3, 3], padding='SAME', activation=lrelu, name='g_conv9_2')(conv9)

#block 10
conv10 = layers.Conv2D(12, [1, 1], padding='SAME', activation=None, name='g_conv10')(conv9)
output = tf.nn.depth_to_space(conv10, 2)

#build model
def SID():
    # return keras.Model(inputs=[input], outputs= [output,pool1, up6,conv6,conv5,conv4])
    return keras.Model(inputs=[input], outputs= output) , "SID3"
# [[[0.08717607 0.07736361 0.04474379]
#   [0.07640272 0.06885926 0.03909069]
#   [0.07248248 0.06613755 0.03975655]
#   ...
#   [0.07979691 0.06711072 0.03508631]
#   [0.08227053 0.0705806  0.04594086]
#   [0.08616085 0.07133029 0.04714175]]
#
#  [[0.09203658 0.07982398 0.05118803]
#   [0.07781867 0.0641927  0.04284708]
#   [0.06979871 0.05801599 0.03476376]
#   ...
#   [0.07782881 0.06234572 0.03340322]
#   [0.08342254 0.06662822 0.04308492]
#   [0.08752189 0.07355367 0.04782175]]
#
#  [[0.08798755 0.07862445 0.04568245]
#   [0.07436459 0.06285917 0.04074244]
#   [0.06788424 0.04736906 0.02739055]
#   ...
#   [0.07322932 0.05753232 0.03216153]
#   [0.08027463 0.06453713 0.04483862]
#   [0.09174325 0.0763857  0.04948162]]
#
#  ...
#
#  [[0.13803148 0.11552387 0.07589895]
#   [0.13914807 0.11658184 0.07819536]
#   [0.15505391 0.12416194 0.07953364]
#   ...
#   [0.18334445 0.15424521 0.10551107]
#   [0.16188811 0.13649723 0.09424594]
#   [0.14503399 0.11639474 0.07811189]]
#
#  [[0.12496267 0.1087539  0.07085502]
#   [0.12619205 0.11083909 0.07286328]
#   [0.13762084 0.11270421 0.07642711]
#   ...
#   [0.15977585 0.13698983 0.09727301]
#   [0.14297327 0.12758541 0.08915487]
#   [0.13209677 0.10898205 0.07271726]]
#
#  [[0.1154258  0.10182866 0.06807604]
#   [0.11462653 0.09924798 0.06955581]
#   [0.12897404 0.10852642 0.07116859]
#   ...
#   [0.1456766  0.12309496 0.08769761]
#   [0.13716431 0.11829633 0.07979646]
#   [0.12081507 0.09997712 0.06739871]]]