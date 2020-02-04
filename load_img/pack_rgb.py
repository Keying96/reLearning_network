#!/usr/bin/env python
# encoding: utf-8
import  os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

input_dir = "../dataset/"
# input_name = "1.bmp"
input_name = "cornsweetBMP.bmp"

def bmp2png(input_dir, input_name, ratio):
    input_path = os.path.join(input_dir, input_name)
    # if input_path.endswith('.bmp') and not input_path.endswith('.png'):
    #     continue
    png_path = input_path[:-4] + ".jpg"

    if not os.path.exists(png_path):
        print ("change dmp to jpg")
        img = Image.open(input_path)
        img = img.convert("RGB")
        img.save(png_path, format = "jpeg", quality = 100)
        img.close()

    input_full = png2rggb(png_path,ratio)
    return input_full

def png2rggb(png_path,ratio=10):
    print ("png path: {}".format(png_path))

    png_arr = plt.imread(png_path)
    png_arr = png_arr.astype(np.float32)
    png_arr = png_arr / 255
    png_shape = png_arr.shape
    H = png_shape[0]
    W = png_shape[1]

    print ("H: {} W:{}".format(H, W))
    remainder_h = H % 4
    remainder_w = W % 4
    im1 = np.expand_dims(png_arr[0:H:2, 0:W:2, 0], axis=2)
    im2 = np.expand_dims(png_arr[0:H:2, 0:W:2, 1], axis=2)
    im3 = np.expand_dims(png_arr[0:H:2, 0:W:2, 2], axis=2)
    im4 = np.expand_dims(png_arr[1:H:2, 1:W:2, 1], axis=2)

    print ("before im1:{}, im2:{}, im3{}, im4{}".format(im1.shape, im2.shape, im3.shape, im4.shape))
    if (remainder_h !=0 or remainder_w !=0):
        im1 = im1[0:-1, 0:-1, :]
        im2 = im2[0:-1, 0:-1, :]
        im3 = im3[0:-1, 0:-1, :]
        print("after im1:{}, im2:{}, im3{}, im4{}".format(im1.shape, im2.shape, im3.shape, im4.shape))

    input_full = np.concatenate((im1, im2, im3, im4), axis=2)
    input_full = np.expand_dims(input_full, axis=0) * ratio

    print (input_full.shape)
    return input_full

if __name__ == '__main__':
    png_path = bmp2png(input_dir, input_name , ratio=10)
    # input_full = png2rggb(png_path)
    print (png_path)
    # print (input_full.shape)
    # print (input_full)