import os
import sys
import dlib
import glob
import imghdr
import numpy as np
from PIL import Image


# use the library https://github.com/davisking/dlib


def alingment(face_file_path, new_path, size=None):

	predictor_path = "C:/Users/virus/Downloads/prueba/shape_predictor_5_face_landmarks.dat"
	#new_path = 'C:/Users/virus/Downloads/prueba/prueba1.jpg'
	#face_file_path = "C:/Users/virus/Downloads/prueba/fear.png"


	# Load all the models we need: a detector to find the faces, a shape predictor
	# to find face landmarks so we can precisely localize the face
	detector = dlib.get_frontal_face_detector()
	sp = dlib.shape_predictor(predictor_path) 

	# Load the image using Dlib
	img = dlib.load_rgb_image(face_file_path)

	# Ask the detector to find the bounding boxes of each face. The 1 in the
	# second argument indicates that we should upsample the image 1 time. This
	# will make everything bigger and allow us to detect more faces.
	dets = detector(img, 1)

	num_faces = len(dets)
	if num_faces == 0:
		print("Sorry, there were no faces found in '{}'".format(face_file_path))
		exit()

	# Find the 5 face landmarks we need to do the alignment.
	faces = dlib.full_object_detections()
	for detection in dets:
		faces.append(sp(img, detection))

	#window = dlib.image_window()

	# Get the aligned face images
	# Optionally: 
	# images = dlib.get_face_chips(img, faces, size=160, padding=0.25)
	#images = dlib.get_face_chips(img, faces, size=320)
	#for image in images:
	#	window.set_image(image)
	#	dlib.hit_enter_to_continue()

	# It is also possible to get a single chip
	if size is None:
		size = 48

	image = dlib.get_face_chip(img, faces[0], size = size)

	#print("TYPE===========>{}".format(new_path))
	
	gr_im = Image.fromarray(image).save(new_path)



def get_folder(string1, string2):
	
	if string1 == string2:
		return None
	
	if  len(string1) < len(string2):
		return string2[len(string1):]
	else:
		return string1[len(string2):]


def face_alingment(path, size):

	if not os.path.isdir(path):
		raise Exception("Invalid directory")
	
	directory = "Outputfile/"
	new_path = os.path.join(path, directory)

	for root, dirs, files in os.walk(path, topdown=False):
		folder = get_folder(root, path)
		if folder is not None:#
			new_folder = os.path.join(new_path, folder)
			if not os.path.isdir(new_path):
				os.mkdir(new_path)
				
			if not os.path.isdir(new_folder):
				print(new_folder)
				os.mkdir(new_folder)
				
			for i, name in enumerate(files, 1):
				original = os.path.join(root+'/', name)
				aux = new_folder + '/' + name

				# face_alingment
				alingment(original, aux, size)
				print("{}".format(len(files) - i))

if __name__ == '__main__':
	
	set_path = 'C:/Users/virus/source/repos/DATASETS/PRUEBA/rotate4.png'
	#'C:/Users/virus/source/repos/FER/datasets/CK+48/'
	size = 48 # size of output image, 48x48

	####face_alingment(set_path, size)
	alingment(set_path, 'C:/Users/virus/source/repos/DATASETS/PRUEBA/rotate1_1.png', 48)