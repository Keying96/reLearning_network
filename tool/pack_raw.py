#!/usr/bin/env python
# -*- coding: utf-8 -*-
import np
import  rawpy

def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out

def covnert_to_channel_first(channel_last_img):
    return channel_fist_img

def load_raw(img_path):
    ratio = 30
    raw = rawpy.imread(img_path)
    input_full = np.expand_dims(pack_raw(raw), axis=0) * ratio
    input_full = np.minimum(input_full, 1.0)

    # # keras 中是CHWN
    # channel_fist_img = covnert_to_channel_first(input_full)

    return  input_full

if __name__ == '__main__':
    img_path = '/home/zhu/PycharmProjects/reLearning_network/dataset/Sony/short/00001_00_0.04s.ARW'
    channel_fist_img = load_raw(img_path)
    print (channel_fist_img)