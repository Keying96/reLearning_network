#!/usr/bin/env python
# encoding: utf-8
import np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import os
from skimage.io import imread, imsave
from skimage.color import rgb2gray, gray2rgb, rgb2lab, lab2rgb
from keras.preprocessing.image import img_to_array, load_img
from skimage import img_as_ubyte
import cv2 as cv
from skimage.util import img_as_float


# the path of input image
decompose_conv_dir = './'
img_path = './lena512.png'
tiff_path = './lena512color.tiff'
img_path = './test.jpg'

lena = cv.imread(tiff_path)
cv.imwrite("lena.jpg", lena)

# #load the input image
# img = rgb2gray(imread(img_path))
# img_h = img.shape[0]
# img_w = img.shape[1]
#
# #create filter
# def gaussian_kernel(size=3, sigma=2):
#     '''
#     size: int,一般取为奇数
#     sigma: blur factor
#
#     return: (normalized) Gaussian kernel，大小 size*size
#     '''
#     x_points = np.arange(-(size - 1) // 2, (size - 1) // 2 + 1, 1)
#     y_points = x_points[::-1]
#     xs, ys = np.meshgrid(x_points, y_points)
#     kernel = np.exp(-(xs ** 2 + ys ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
#     kernel = kernel / kernel.sum()
#     return kernel
#
def sobel_kernel():
    kernel = tf.constant([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
    ], tf.float64)
    return kernel
#
def perwitt_kernel():
    kernel = tf.constant([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ],tf.float64)
    return kernel


# #convolution
# input = tf.reshape(img,[1,img_h,img_w,1])
#
kernel = perwitt_kernel()
# kernel = sobel_kernel()
# kernel = tf.reshape(kernel, [3,3,1,1])
# conv = tf.nn.conv2d(input, kernel,[1,1,1,1],padding = "SAME")
#
# #save the output of convolution
# conv_img = conv.numpy()[0][:,:,0]
# plt.imshow(conv_img,cmap='gray')
# plt.axis('off')
# savename = os.path.join(decompose_conv_dir, '{}.png'.format("conv"))
# plt.savefig(savename, dpi=600)
# plt.close()
#
# # # print (conv.numpy()[0][:,:,0].shape)
# # # conv = gray2rgb(conv_img)
# # # scipy.misc.imsave("conv.tif", conv_img)
#
plt.imshow(kernel,cmap='gray')
plt.axis('off')
savename = os.path.join(decompose_conv_dir, '{}.png'.format("per"))
plt.savefig(savename, dpi=600)
plt.close()

