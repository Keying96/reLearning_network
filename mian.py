#!/usr/bin/env python
# encoding: utf-8
import numpy as np
import tensorflow as tf
import os,re
from network import SID
from tool import pack_raw as pack_raw
from tensorflow.python import pywrap_tensorflow

input_path = "./dataset/Sony/short/00001_00_0.04s.ARW"
sony_checkpoint_dir = "./checkpoint/Sony/"
layer_name_list = []

__re_digits = re.compile(r'(\d+)')

def __emb_numbers(s):
    pieces = __re_digits.split(s)
    pieces[1::2] = map(int, pieces[1::2])
    return pieces


def __sort_strings_with_emb_numbers(alist):
    aux = [(__emb_numbers(s), s) for s in alist]
    aux.sort()
    return [s for __, s in aux]

#build model
model = SID.SID()

#load input image
input_full = pack_raw.load_raw(input_path)

#load weight
sony_checkpoint_dir = os.path.join(sony_checkpoint_dir, "model.ckpt")
reader = pywrap_tensorflow.NewCheckpointReader(sony_checkpoint_dir)  # tf.train.NewCheckpointReader
var_to_shape_map = reader.get_variable_to_shape_map()

# model_layer={}
#
# for key in var_to_shape_map:
#     print ("tensor name: {}".format(key))
#     if not key in model_layer.keys():
#         model_layer[key]=[reader.get_tensor(key)]
#     else:
#         model_layer[key].append(reader.get_tensor(key))
#
# np.save('Sony_model.npy',model_layer)

for key in var_to_shape_map:
    if "/weights/Adam_1" in key:
        layer_name_list.append(key.split("/weights/Adam_1")[0])


for layer_name in layer_name_list:
    layer = model.get_layer(layer_name)
    key_kernel = layer_name + "/weights"
    key_biases = layer_name + "/biases"

    weights = reader.get_tensor(key_kernel)
    weights = weights[0,:,:,:]
    # print (weights)
    biases = reader.get_tensor(key_biases)
    print ("the length of weights: {}".format(len(weights)))
    print ("the length of biases: {}".format(len(biases)))

    layer.set_weights(weights)
    layer.set_biases(biases = biases)
    # print ("{} {}".format(key_kernel, weights))
    # layer.set_weights(reader.get_tensor(key_kernel))



#run
output = model(input_full)