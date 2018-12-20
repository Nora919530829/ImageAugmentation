import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

IMAGE_SIZE = 224

def tf_resize_images(x_img_file_paths):
	x_data = []
	tf.reset_default_graph()
	x = tf.placeholder(tf.float32, [None,None,3])
	tf_img = tf.image.resize_images(x,(IMAGE_SIZE,IMAGE_SIZE),tf.image.ResizeMethod.NEAREST_NEIGHBOR)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		# Each image is resized individually as different image may be of different size
		for index, file_path in enumerate(x_img_file_paths):
			img = mpimg.imread(file_path)
			resized_img = sess.run(tf_img,feed_dict={x:img})
			x_data.append(resized_img)

	x_data = np.array(x_data,dtype=np.float32)    # convert to numpy

	matplotlib.rcParams.update({'font.size':14})
	fig, ax = plt.subplots(figsize = (12, 12))
	plt.subplot(1, 2, 1)
	plt.imshow(mpimg.imread(x_img_file_paths[0])[:,:,:3])
	plt.title('Original Image')
	plt.subplot(1, 2, 2)
	plt.imshow(x_data[0])
	plt.title('Resized Image')
	plt.show()

	return x_data

