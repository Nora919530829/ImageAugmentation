import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import numpy as np
import os
import cv2
from math import floor, ceil, pi

from get_image_paths import get_image_paths
from tf_resize_images import tf_resize_images
from display_all_images import display_all_images
from Image_Augmentation import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('IMAGE_SIZE', 224,'IMAGE_SIZE')

def main(argv=None):
	x_img_paths = get_image_paths()
	print(x_img_paths)

	x_imgs = tf_resize_images(x_img_paths)
	print(x_imgs.shape)

	display_all_images(x_imgs)

	# Image Augmentation Techniques:

	# 1. Scaling

	# Produce each image at scaling of 90%, 75% and 60% of original image.
	scaled_imgs = central_scale_images(x_imgs, [0.90, 0.75, 0.60])
	print(scaled_imgs.shape)
	fig, ax = plt.subplots(figsize = (10, 10))
	plt.subplot(2, 2, 1)
	plt.imshow(x_imgs[1])
	plt.title('Base Image')
	plt.subplot(2, 2, 2)
	plt.imshow(scaled_imgs[3])
	plt.title('Scale = 0.90')
	plt.subplot(2, 2, 3)
	plt.imshow(scaled_imgs[4])
	plt.title('Scale = 0.75')
	plt.subplot(2, 2, 4)
	plt.imshow(scaled_imgs[5])
	plt.title('Scale = 0.60')
	plt.show()

	# 2. translation
	translated_imgs = translate_images(x_imgs)
	print(translated_imgs.shape)

	gs = gridspec.GridSpec(1, 5)
	gs.update(wspace = 0.30, hspace = 2)

	fig, ax = plt.subplots(figsize = (16, 16))
	plt.subplot(gs[0])
	plt.imshow(x_imgs[2])
	plt.title('Base Image')
	plt.subplot(gs[1])
	plt.imshow(translated_imgs[2])
	plt.title('Left 20 percent')
	plt.subplot(gs[2])
	plt.imshow(translated_imgs[14])
	plt.title('Right 20 percent')
	plt.subplot(gs[3])
	plt.imshow(translated_imgs[26])
	plt.title('Top 20 percent')
	plt.subplot(gs[4])
	plt.imshow(translated_imgs[38])
	plt.title('Bottom 20 percent')
	plt.show()

	gs = gridspec.GridSpec(1, 5)
	gs.update(wspace = 0.30, hspace = 2)

	fig, ax = plt.subplots(figsize = (16, 16))
	plt.subplot(gs[0])
	plt.imshow(x_imgs[3])
	plt.title('Base Image')
	plt.subplot(gs[1])
	plt.imshow(translated_imgs[3])
	plt.title('Left 20 percent')
	plt.subplot(gs[2])
	plt.imshow(translated_imgs[15])
	plt.title('Right 20 percent')
	plt.subplot(gs[3])
	plt.imshow(translated_imgs[27])
	plt.title('Top 20 percent')
	plt.subplot(gs[4])
	plt.imshow(translated_imgs[39])
	plt.title('Bottom 20 percent')
	plt.show()

	# 3. rotate
	rotated_imgs = rotate_images1(x_imgs)
	print(rotated_imgs.shape)

	fig, ax = plt.subplots(figsize = (10, 10))
	plt.subplot(2, 2, 1)
	plt.imshow(x_imgs[4])
	plt.title('Base Image')
	plt.subplot(2, 2, 2)
	plt.imshow(rotated_imgs[12])
	plt.title('Rotate 90 degrees')
	plt.subplot(2, 2, 3)
	plt.imshow(rotated_imgs[13])
	plt.title('Rotate 180 degrees')
	plt.subplot(2, 2, 4)
	plt.imshow(rotated_imgs[14])
	plt.title('Rotate 270 degrees')
	plt.show()

	# 4. rotate at any angle
	# Start rotation at -90 degrees, end at 90 degrees and produce totally 14 images
	rotated_imgs = rotate_images2(x_imgs, -90, 90, 14)
	print(rotated_imgs.shape)
	matplotlib.rcParams.update({'font.size': 11})

	fig, ax = plt.subplots(figsize = (16, 16))
	gs = gridspec.GridSpec(3, 5)
	gs.update(wspace = 0.30, hspace = 0.0002)

	plt.subplot(gs[0])
	plt.imshow(x_imgs[5])
	plt.title('Base Image')

	for i in range(14):
		plt.subplot(gs[i + 1])
		plt.imshow(rotated_imgs[5 + 12 * i])
		plt.title('Rotate {:.2f} degrees'.format(-90 + 13.846 * i))
	plt.show()

	# 5. flip
	flipped_images = flip_images(x_imgs)
	print(flipped_images.shape)

	matplotlib.rcParams.update({'font.size': 14})

	fig, ax = plt.subplots(figsize = (10, 10))
	plt.subplot(2, 2, 1)
	plt.imshow(x_imgs[6])
	plt.title('Base Image')
	plt.subplot(2, 2, 2)
	plt.imshow(flipped_images[18])
	plt.title('Flip left right')
	plt.subplot(2, 2, 3)
	plt.imshow(flipped_images[19])
	plt.title('Flip up down')
	plt.subplot(2, 2, 4)
	plt.imshow(flipped_images[20])
	plt.title('Transpose')
	plt.show()

	# 6. salt_pepper_noise
	salt_pepper_noise_imgs = add_salt_pepper_noise(x_imgs)
	print(salt_pepper_noise_imgs.shape)
	
	fig, ax = plt.subplots(figsize = (10, 10))
	plt.subplot(2, 2, 1)
	plt.imshow(x_imgs[7])
	plt.title('Base Image')
	plt.subplot(2, 2, 2)
	plt.imshow(salt_pepper_noise_imgs[7])
	plt.title('Salt pepper noise image')
	plt.subplot(2, 2, 3)
	plt.imshow(x_imgs[8])
	plt.title('Base Image')
	plt.subplot(2, 2, 4)
	plt.imshow(salt_pepper_noise_imgs[8])
	plt.title('Salt pepper noise image')
	plt.show()

	# 7. Lighting Condition
	gaussian_noise_imgs = add_gaussian_noise(x_imgs)
	print(gaussian_noise_imgs.shape)

	fig, ax = plt.subplots(figsize = (10, 10))
	plt.subplot(2, 2, 1)
	plt.imshow(x_imgs[9])
	plt.title('Base Image')
	plt.subplot(2, 2, 2)
	plt.imshow(gaussian_noise_imgs[9])
	plt.title('Shaded image')
	plt.subplot(2, 2, 3)
	plt.imshow(x_imgs[10])
	plt.title('Base Image')
	plt.subplot(2, 2, 4)
	plt.imshow(gaussian_noise_imgs[10])
	plt.title('Shaded image')
	plt.show()

	# 8.Perspective Transform
	x_img = x_imgs[11]
	perspective_img = perspective_transform(x_img)
	print(perspective_img.shape)

	fig, ax = plt.subplots(figsize = (12, 12))
	plt.subplot(1, 2, 1)
	plt.imshow(x_imgs[11])
	plt.title('Original Image')
	plt.subplot(1, 2, 2)
	plt.imshow(perspective_img)
	plt.title('Different View of Image')
	plt.show()



if __name__ == '__main__':
	tf.app.run()
	
