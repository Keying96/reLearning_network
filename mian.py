#!/usr/bin/env python
# encoding: utf-8
import numpy as np
from load_img import pack_rgb
import os,re,glob
from tool import pack_raw as pack_raw
from tensorflow.python import pywrap_tensorflow
from PIL import Image
import matplotlib.pyplot as plt
import time
from network import SID
from network import tool
# from network import resieze_SID
# from network import  resize_SIDto3


#input image path
input_dir = "./dataset/"
# in_name = "1.bmp"
# in_name = "fkdjn.jpg"
# in_name = "gra.jpg"
# in_name = "circles24.jpg"
# in_name = "LumaZonePlate.png"
# in_name = "rings.png"
# in_name = "Zone_Plate_Bicubic.png"
# in_name = "cornsweetBMP.bmp"

# input_dir = "./dataset/Sony/short/"
gt_dir = "./dataset/Sony/long/"
# in_name = "00001_00_0.1s.ARW"
# in_name = "00001_00_0.04s.ARW"
in_name = "00014_00_10s.ARW"
# input_dir = "./dataset/"
# in_name = "BRTB7511.DNG"
# in_name = "bmp2raw.raw"
# in_name = 'noise2.bmp'
#checkpoint file path
sony_checkpoint_dir = "./checkpoint/Sony/"
result_dir = './reduce_layer'
processing_dir = './process'


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

    # # load input image
    # # input full is NHWC
    # # input image is short exposure time
    # input_full = pack_raw.load_raw(in_name, gt_dir, input_dir)

    #
    # ratio_list = [1, 10, 20 ,30]
    # for i in range(len(ratio_list)):
    #     input_full = pack_raw.load_raw2(in_name,input_dir, ratio=ratio_list[i])
    # input_full = pack_raw.load_raw2(in_name,input_dir, ratio=10)
    # input_full =  pack_rgb.bmp2png(input_dir, in_name , ratio=ratio_list[i])

    input_full = pack_raw.load_raw2(in_name,gt_dir, ratio= 1)
    input_shape = input_full.shape

    reader, var_to_shape_map = get_node(sony_checkpoint_dir)
    # build model
    model, model_name = SID.SID()
    # model, model_name = resieze_SID.SID()
    # model, model_name = resize_SIDto3.SID()
    print(model.summary())

    # load weight
    load_weight(reader, var_to_shape_map, model)

    # output, pool1, up6,conv6,conv5,conv4 = model(input_full)
    star_time = time.time()
    output, conv1 = model(input_full)
    end_time = time.time()
    print ("the runtime is {}".format(end_time-star_time))

    #original runtime 4.222529649734497
    #rewrite runtime 2.8887720108032227
    #rewriteto3 runtime 1.9794270992279053
    #输出中间过程图
    # tool.write_img(conv1, processing_dir, model_name, in_name)
    # tool.write_img_ratio(conv1, processing_dir, model_name, in_name, ratio=10)
    # output = np.minimum(np.maximum(output, 0), 1)
    output = output[0, :, :, :]
    # print (output)

    # #plt图像输出
    plt.axis("off")
    plt.imshow(output)
    # savename = os.path.join(result_dir, 'whole_image_rewriteto3.png')
    savename = os.path.join(result_dir, '{}_{}'.format(os.path.splitext(in_name)[0], model_name) + '.png')
    # savename = os.path.join(result_dir, '{}_{}_{}'.format(os.path.splitext(in_name)[0], model_name, ratio_list[i]) + '.jpg')
    # savename = os.path.join(result_dir, '{}_{}_{}'.format(os.path.splitext(in_name)[0], model_name, 10) + '.png')



    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    plt.savefig(savename, dpi=600)
    print(savename)
    plt.close()
