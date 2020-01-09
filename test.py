#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python import pywrap_tensorflow
import os,re
# from network import SID

checkpoint_dir = './checkpoint/Sony/'

# print (sess.run(deconv_filter))
layer_name_list = []
variable_model_name_list = ["tf_op_layer_Shape", "tf_op_layer_Shape_1","tf_op_layer_Shape_2","tf_op_layer_Shape_3"]

__re_digits = re.compile(r'(\d+)')

def __emb_numbers(s):
    pieces = __re_digits.split(s)
    pieces[1::2] = map(int, pieces[1::2])
    return pieces


def __sort_strings_with_emb_numbers(alist):
    aux = [(__emb_numbers(s), s) for s in alist]
    aux.sort()
    return [s for __, s in aux]

#load weight
def get_node(sony_checkpoint_dir):
    sony_checkpoint_dir = os.path.join(sony_checkpoint_dir, "model.ckpt")
    reader = pywrap_tensorflow.NewCheckpointReader(sony_checkpoint_dir)  # tf.train.NewCheckpointReader
    var_to_shape_map = reader.get_variable_to_shape_map()

    return reader,var_to_shape_map


def load_weight(reader, var_to_shape_map,deconv_filter_name,x1_name,x2_name):
    x1_name = x1_name + "/weights"
    x2_name = x2_name + "/weights"

    layer_name_list = []
    variable_name_list = []
    other_name_list = []
    # reader, var_to_shape_map = get_node(sony_checkpoint_dir)

    # get the layer name list
    # 可以通过resize 网络节点获取name_list,这样可以动态获取layer names
    for key in var_to_shape_map:
        if "/weights/Adam_1" in key:
            layer_name_list.append(key.split("/weights/Adam_1")[0])
        elif "Variable" in key:
            variable_name_list.append(key)
        else:
            other_name_list.append(key)

    layer_name_list = __sort_strings_with_emb_numbers(layer_name_list)
    variable_name_list = __sort_strings_with_emb_numbers(variable_name_list)
    print("other_name_list:{}".format(other_name_list))
    print ("variable name list:{}".format(variable_name_list))

    # # set the weights of each layer
    # for layer_name in layer_name_list:
    #     input_list = []
    #
    #     try:
    #         layer = model.get_layer(layer_name)
    #         key_kernel = layer_name + "/weights"
    #         key_biases = layer_name + "/biases"
    #
    #         weights = reader.get_tensor(key_kernel)
    #         biases = reader.get_tensor(key_biases)
    #         input_list.append(weights)
    #         input_list.append(biases)
    #
    #         layer.set_weights(input_list)
    #
    #     except:
    #         pass
    #
    #     # set the weights of each layer
    #     for i in range(len(variable_model_name_list)):
    #
    #         layer = model.get_layer(variable_model_name_list[i])
    #         # print ("variable_model_name_list[i]:{}".format(variable_model_name_list[i]))
    #         # print (layer.get_weights())
    #
    #         key_kernel = variable_name_list[i]
    #
    #         weights = reader.get_tensor(key_kernel)
    #         # print(weights.shape)
    #         # print (weights)
    return  reader.get_tensor(deconv_filter_name)\
        ,reader.get_tensor(x1_name)\
        ,reader.get_tensor(x2_name)



if __name__ == '__main__':

    pool_size = 2
    output_channels = 256
    in_channels = 512

    deconv_filter_name = "Variable"
    x1_name = "g_conv5_2"
    x2_name = "g_conv4_2"
    strides = [1, 2, 2, 1]

    reader, var_to_shape_map = get_node(checkpoint_dir)
    # model = SID.SID()
    deconv, x1, x2 = load_weight(reader, var_to_shape_map,
                          deconv_filter_name,x1_name,x2_name)
    x1_shape = x1.shape
    x2_shape = x2.shape
    print (deconv.shape)
    print (x1.shape)
    print (x2.shape)

    b = tf.constant(0,shape=[output_channels])

    # x1 = tf.reshape(x1,x2.shape)
    # expected_l = tf.nn.conv2d(x1, deconv, strides=strides, padding='SAME')
    # expected_l = layers.Conv2D(256, 2, padding='SAME', weights = [deconv,b],name='deconv')(x1)
    # expected_l = layers.Conv2D(filters = output_channels, kernel_size = pool_size, strides=(1, 1),
    #                                     padding='same', weights= [deconv,b],name = "Variable")(x1)
    expected_l = layers.Conv2DTranspose(filters = output_channels, kernel_size = pool_size, strides=(1, 1),
                                        padding='same', weights= [deconv,b],name = "Variable")(x1)
    # expected_l = tf.nn.conv2d_transpose(x1, deconv, tf.shape(x2), strides=[1, pool_size, pool_size, 1])
    print(expected_l)

    # deconv_output = layers.concatenate([expected_l,x2], axis = 3)
    # deconv_output = tf.concat([expected_l, x2], 2)
    # deconv_output.set_shape([None, None, None, output_channels * 2])
    # print (deconv_output)