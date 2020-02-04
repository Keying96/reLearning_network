#!/usr/bin/env python
# -*- coding: utf-8 -*-
import np,os
import  rawpy,glob

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
    print("out size: {}".format(out.shape))
    return out

def covnert_to_channel_first(channel_last_img):
    return channel_fist_img

# get test ratio
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
    return ratio ,in_path

def load_raw(in_name, gt_dir, input_dir):
    ratio, img_path, = get_ratio(in_name, gt_dir, input_dir)

    raw = rawpy.imread(img_path)
    input_full = np.expand_dims(pack_raw(raw), axis=0) * ratio
    input_full = np.minimum(input_full, 1.0)

    # # keras 中是CHWN
    # channel_fist_img = covnert_to_channel_first(input_full)

    return  input_full

def load_raw2(in_name, input_dir, ratio):
    img_path = os.path.join(input_dir, in_name)
    raw = rawpy.imread(img_path)
    input_full = np.expand_dims(pack_raw(raw), axis=0) * ratio
    input_full = np.minimum(input_full, 1.0)

    # # keras 中是CHWN
    # channel_fist_img = covnert_to_channel_first(input_full)

    return  input_full

if __name__ == '__main__':
    # img_path = '/home/zhu/PycharmProjects/reLearning_network/dataset/Sony/short/00001_00_0.04s.ARW'
    img_path = "/home/zhu/PycharmProjects/reLearning_network/dataset/bmp2raw.raw"
    # channel_fist_img = load_raw(img_path)
    # print (channel_fist_img)
    raw = rawpy.imread(img_path)
    out = pack_raw(raw)
    print (out)
