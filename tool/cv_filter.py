#!/usr/bin/env python
# encoding: utf-8
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from network import tool

save_bulr_dir = './bulr/'
save_sobel_dir = './sobel/'
save_laplacian_dir = './laplacian/'
save_perwitt = './perwitt/'
tool.prepare_dir(save_bulr_dir)
tool.prepare_dir(save_sobel_dir)
tool.prepare_dir(save_laplacian_dir)
tool.prepare_dir(save_perwitt)

img = cv2.imread('./lena512.bmp')


# kernel = np.ones((5,5),np.float32)/25
# kernel_blur = np.array((
#     [0.0625, 0.125, 0.0625],
#     [0.125, 0.25, 0.125],
#     [0.0625, 0.125, 0.0625]), dtype="float32")

# kernel_sobel = np.array((
#     [-1, -2, -1],
#     [0, 0, 0],
#     [1, 2, 1]), dtype="float32")

# kernel_laplacian = np.array((
#     [0, 1, 0],
#     [1, -4, 1],
#     [0, 1, 0]), dtype="float32")

kernel_perwitt = np.array((
            [-1,0,1],
            [-1,0,1],
            [-1,0,1]), dtype="float32")


# dst = cv2.filter2D(img,-1,kernel_perwitt)

plt.set_cmap('gray')
# savename_input = os.path.join(save_perwitt,"{}.png".format("input"))
# savename_output = os.path.join(save_perwitt,"{}.png".format("result"))
savename_kernel = os.path.join(save_perwitt,"{}.png".format("kernel"))
# plt.imsave(savename_input, img)
# plt.imsave(savename_output, dst)
# cv2.resize(kernel_perwitt,(100,100),'nearest');
plt.imsave(savename_kernel, kernel_perwitt, dpi = 600)
plt.close()

ori = cv2.imread(savename_kernel)
height, width = ori.shape[:2]
after = cv2.resize(ori,(100*width, 100*height),interpolation = cv2.INTER_LINEAR)
plt.imsave(savename_kernel, after, dpi = 600)
plt.close()

# plt.subplot(121),plt.imshow(img),plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
# plt.xticks([]), plt.yticks([])