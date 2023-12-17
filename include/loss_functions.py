# -*- coding: utf-8 -*-
"""
Updated at Dec 16 2023

@author: Alireza Norouzi
source : https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/losses/dssim.py
"""

from __future__ import absolute_import
import tensorflow as tf
import numpy as np
keras = tf.keras

class SSIM_MSE_LOSS():
	def __init__(self, ssim_relative_loss, mse_relative_loss, ssim_win_size=4):
		self.ssim_relative_loss = tf.convert_to_tensor(ssim_relative_loss/(ssim_relative_loss+mse_relative_loss), tf.float32)
		self.mse_relative_loss = tf.convert_to_tensor(mse_relative_loss / (ssim_relative_loss+mse_relative_loss), tf.float32)
		self.win_size = ssim_win_size
		
	def ssimmse_loss(self, y_true, y_pred):
		return 1.0 - (self.ssim_relative_loss)*(self.tf_ssim(y_true, y_pred, size=self.win_size)) + (self.mse_relative_loss)*(tf.losses.mean_squared_error(y_true, y_pred))


	def tf_ssim(self, img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
		window = _tf_fspecial_gauss(size, sigma) # window shape [size, size]
		K1 = 0.01
		K2 = 0.03
		L = 1  # depth of image (255 in case the image has a differnt scale)
		C1 = (K1*L)**2
		C2 = (K2*L)**2
		mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID')
		mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='VALID')
		mu1_sq = mu1*mu1
		mu2_sq = mu2*mu2
		mu1_mu2 = mu1*mu2
		sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='VALID') - mu1_sq
		sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='VALID') - mu2_sq
		sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='VALID') - mu1_mu2
		if cs_map:
			value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
						(sigma1_sq + sigma2_sq + C2)),
					(2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
		else:
			value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
						(sigma1_sq + sigma2_sq + C2))

		if mean_metric:
			value = tf.reduce_mean(value)
		return value

def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)
