#!/usr/bin/env python
# encoding: utf-8
import  os, errno
import numpy as np
import imageio
import types

def prepare_dir(path, empty=False):
    """
    Creates a directory if it soes not exist
    :param path: string, path to desired directory
    :param empty: boolean, delete all directory content if it exists
    :return: nothing
    """

    def create_dir(path):
        """
        Creates a directory
        :param path: string
        :return: nothing
        """
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

    if not os.path.exists(path):
        create_dir(path)

def write_img(rgb_array, output_dir,model_name, in_name):
    prepare_dir(output_dir)
    rgb_array = np.squeeze(rgb_array)
    print (rgb_array.shape[2])
    for i in range(rgb_array.shape[2]):
        outimg = rgb_array[:,:,i].copy()
        outimg = np.minimum(outimg, 1.0)
        outimg = outimg * 255
        img_name = os.path.splitext(in_name)[0]
        outname  = img_name + "_" + model_name + "_" +str(i) + ".png"
        ouput_path = os.path.join(output_dir, outname)
        print ("ouput_path: {}".format(ouput_path))
        imageio.imwrite(ouput_path, outimg.astype("uint8"))

def write_img_ratio(rgb_array, output_dir,model_name, in_name, ratio):
    prepare_dir(output_dir)
    img_name = os.path.splitext(in_name)[0]
    file_name = img_name + "_" + model_name + "_" + str(ratio)
    file_path = os.path.join(output_dir, file_name)
    prepare_dir(file_path)
    rgb_array = np.squeeze(rgb_array)
    print (rgb_array.shape[2])
    for i in range(rgb_array.shape[2]):
        outimg = rgb_array[:,:,i].copy()
        outimg = np.minimum(outimg, 1.0)
        outimg = outimg * 255
        outname  = img_name + "_" + model_name + "_" +str(i) + ".png"
        ouput_path = os.path.join(file_path, outname)
        print ("ouput_path: {}".format(ouput_path))
        imageio.imwrite(ouput_path, outimg.astype("uint8"))
