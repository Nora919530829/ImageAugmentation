import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import numpy as np
import os
import cv2
from math import floor, ceil, pi

FLAGS = tf.app.flags.FLAGS

def display_all_images(x_imgs, n_cols = 4):
	n_rows = ceil(len(x_imgs)/n_cols)
	display_image = np.zeros([n_rows * FLAGS.IMAGE_SIZE, n_cols * FLAGS.IMAGE_SIZE, 3],dtype = np.float32)
	for i in range(n_rows):
		for j in range(n_cols):
			x_img = x_imgs[i*n_cols+j]
			disp_padded = np.pad(x_img, ((i * FLAGS.IMAGE_SIZE, (n_rows - 1 - i) * FLAGS.IMAGE_SIZE),
                                         (j * FLAGS.IMAGE_SIZE, (n_cols - 1 - j) * FLAGS.IMAGE_SIZE), (0, 0)), 'constant')
			display_image = np.add(display_image, disp_padded)

	plt.figure(figsize = (n_rows * 3, n_cols * 3))
	plt.imshow(display_image)
	plt.axis('off')
	plt.show()
