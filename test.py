#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
from network import network as network

checkpoint_dir = "./checkpoint/Sony/checkpoint"
model = network.network(256)  # 输入图像尺寸的最小值：256
checkpoint = tf.train.Checkpoint(model=model)
# checkpoint.restore(checkpoint_dir)
# print(tf.train.list_variables(checkpoint_dir))

model.load_weights(checkpoint_dir)
# out = model.predict(img)

# # saver=tf.train.import_meta_graph('./checkpoint/Sony/model.ckpt.meta')
# # ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
# tf.train.list_variables(tf.train.latest_checkpoint(checkpoint_dir))
#
# sess = tf.Session()
# in_image = tf.placeholder(tf.float32, [None, None, None, 4])
# gt_image = tf.placeholder(tf.float32, [None, None, None, 3])
# out_image = network(in_image)
#
# saver = tf.train.Saver()
# sess.run(tf.global_variables_initializer())
# ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
# if ckpt:
#     print('loaded ' + ckpt.model_checkpoint_path)
#     saver.restore(sess, ckpt.model_checkpoint_path)
#
# if not os.path.isdir(result_dir + 'final/'):
#     os.makedirs(result_dir + 'final/')
#
# # test the first image in each sequence
# in_files = glob.glob(input_dir + '%05d_00*.ARW' % test_id)
# in_path = in_files[k]
# in_fn = os.path.basename(in_path)
# print(in_fn)
#
# ratio = 300
#
# raw = rawpy.imread(in_path)
# input_full = np.expand_dims(pack_raw(raw), axis=0) * ratio
#
# im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
# # scale_full = np.expand_dims(np.float32(im/65535.0),axis = 0)*ratio
# scale_full = np.expand_dims(np.float32(im / 65535.0), axis=0)
#
# im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
# gt_full = np.expand_dims(np.float32(im / 65535.0), axis=0)
#
# input_full = np.minimum(input_full, 1.0)
#
# output = sess.run(out_image, feed_dict={in_image: input_full})
# output = np.minimum(np.maximum(output, 0), 1)
#
# output = output[0, :, :, :]
# scipy.misc.toimage(output * 255, high=255, low=0, cmin=0, cmax=255).save(
#     result_dir + 'final/%5d_00_%d_out.png' % (test_id, ratio))