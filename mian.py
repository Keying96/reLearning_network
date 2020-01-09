#!/usr/bin/env python
# encoding: utf-8
import numpy as np
import os,re
# from network import SID
from network import resieze_SID
from tool import pack_raw as pack_raw
from tensorflow.python import pywrap_tensorflow
from PIL import Image
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from network.UpSampleConcat import UpSampleConcat as UpSampleConcat


#input image path
input_path = "./dataset/Sony/short/00001_00_0.1s.ARW"

#checkpoint file path
sony_checkpoint_dir = "./checkpoint/Sony/"

result_dir = './reduce_layer'


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


def load_weight(reader, var_to_shape_map, model):
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
    print("variable_name_list:{}".format(variable_name_list))


    # set the weights of each layer
    for layer_name in layer_name_list:
        input_list = []

        try:
            layer = model.get_layer(layer_name)
            key_kernel = layer_name + "/weights"
            key_biases = layer_name + "/biases"

            weights = reader.get_tensor(key_kernel)
            biases = reader.get_tensor(key_biases)
            input_list.append(weights)
            input_list.append(biases)

            layer.set_weights(input_list)

        except:
            pass

        for variable_name in variable_name_list:
            try:
                layer = model.get_layer(variable_name)
                weights = reader.get_tensor(variable_name)
                layer.set_weights([weights,])

            except:
                pass



if __name__ == '__main__':

    # load input image
    # input full is NHWC
    input_full = pack_raw.load_raw(input_path)
    input_shape = input_full.shape

    reader, var_to_shape_map = get_node(sony_checkpoint_dir)
    # build model
    # model = SID.SID()
    model = resieze_SID.SID()
    print(model.summary())

    # load weight
    load_weight(reader, var_to_shape_map, model)

    # output, pool1, up6,conv6,conv5,conv4 = model(input_full)
    star_time = time.time()
    output = model(input_full)
    end_time = time.time()
    print ("the runtime is {}".format(end_time-star_time))

    #original runtime 4.222529649734497
    #rewrite runtime 2.8887720108032227
    output = np.minimum(np.maximum(output, 0), 1)
    output = output[0, :, :, :]
    print (output)

    # #图像输出
    plt.axis("off")
    plt.imshow(output)
    savename = os.path.join(result_dir, 'whole_image_rewrite.png')  # the runtime is 5.30000114441

    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    plt.savefig(savename, dpi=600)
    print(savename)
    plt.close()
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