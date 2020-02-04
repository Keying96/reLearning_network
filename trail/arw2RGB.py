#!/usr/bin/env python
# encoding: utf-8
import  os, glob, rawpy
from network import tool as tool
import numpy as  np
import imageio

gt_dir = "../dataset/Sony/long/"
input_dir = "../dataset/Sony/short/"
output_dir = "../dataset/RGB"
input_name = "00001_00_0.04s.ARW"

def get_ratio(in_name, gt_dir, input_dir):
    test_id = int(os.path.basename(in_name)[0:5])
    gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % test_id)
    gt_path = gt_files[0]
    gt_fn = os.path.basename(gt_path)
    in_path = os.path.join(input_dir, in_name)
    print ("in_path:{}".format(in_path))

    in_fn = os.path.basename(in_path)
    in_exposure = float(in_fn[9:-5])
    gt_exposure = float(gt_fn[9:-5])
    ratio = min(gt_exposure / in_exposure, 300)
    return ratio

def raw_process(input_path, ration):
    # pack Bayer image to 4 channels
    raw = rawpy.imread(input_path)
    im= raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    dms_img = np.zeros((H//2, W//2, 3))
    # Red
    dms_img[:,:,0] = np.squeeze(im[0:H:2, 0:W:2, :])
    #Green
    dms_img[:,:,1] = np.squeeze((im[0:H:2, 1:W:2, :] + im[1:H:2, 1:W:2, :]) / 2)
    #Blue
    dms_img[:,:,2] = np.squeeze(im[1:H:2, 0:W:2, :])

    # input_full = np.expand_dims(dms_img, axis=0) * ratio
    input_full = dms_img * ratio
    input_full = np.minimum(input_full, 1.0)

    return input_full

def write(rgb_array, output_path):
    print (rgb_array.shape)
    outimg = rgb_array.copy()
    # outimg[outimg < 0] = 0
    outimg = outimg * 255
    imageio.imwrite(output_path, outimg.astype('uint8'))


if __name__ == '__main__':
    tool.prepare_dir(output_dir)

    input_path = os.path.join(input_dir, input_name)
    out_name = os.path.splitext(input_name)[0] + "_rewrite" + ".png"
    output_path = os.path.join(output_dir, out_name)
    print ("output_path:{}".format(output_path))

    ratio = get_ratio(input_name, gt_dir, input_dir)
    rgb_array = raw_process(input_path, ratio)
    write (rgb_array,output_path)

