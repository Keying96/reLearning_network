#!/usr/bin/env python
# encoding: utf-8

from network import network as network
from tool import pack_raw as pack_raw
import tensorflow as tf
img_path = './dataset/Sony/short/00001_00_0.04s.ARW'
sony_checkpoint_dir = "./checkpoint/Sony/"

# print (tf.train.list_variables(tf.train.latest_checkpoint(sony_checkpoint_dir)))


#加载图像
# input_full = pack_raw.load_raw(img_path)
# print (input_full.shape)
# img_h = input_full.shape[1]
# img_w = input_full.shape[2]
# img_c = input_full.shape[3]
# input_full = tf.ones(shape= (1,1424,2428,1))


# #加载模型参数
# network = network.network(img_h, img_w, img_c)
# network = network.network(input_full)
# network = network.network(input_full)
network = network.network()
# network = tf.Keras.models(inputs=input, outputs=output)
model = tf.train.Checkpoint(model = network)
model.restore(tf.train.latest_checkpoint(sony_checkpoint_dir))
# test_out = model(input_full, training = False
print(tf.train.list_variables(tf.train.latest_checkpoint(sony_checkpoint_dir)))

