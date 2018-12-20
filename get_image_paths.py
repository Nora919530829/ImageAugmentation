import tensorflow as tf
import numpy as np
import os
import cv2

def get_image_paths():
	folder = './RawImages'
	files = os.listdir(folder)
	files.sort()
	files = ['{}/{}'.format(folder,file) for file in files]
	return files

