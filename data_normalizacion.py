import os
import cv2
import numpy as np
from PIL import Image
from scipy import stats
from numpy import asarray
from PIL import Image, ImageDraw


def downsampling_apply(image, new_folder, size = 32):
	
	 # or whatever you are doing to the image
	
	try:
		#Processes the image
		proc_image = Image.open(image).convert("L")
		
		proc_image = proc_image.resize((size, size), Image.ANTIALIAS)
		
		proc_image.save(new_folder)

	except Exception:
		print("An error has ocurred while trying to save a resized image...")


# histogram equalization and z-score normalizacion
def histogram_equ(path):


	if not os.path.isdir(path):
		raise Exception("Invalid directory")
	
	directory = "HistogramApplied/"
	new_path = os.path.join(path, directory)

	for root, dirs, files in os.walk(path, topdown=False):

		if not os.path.isdir(new_path):
			os.mkdir(new_path)

		for i, name in enumerate(files, 1):
			
			original = os.path.join(root, name)
			output_path = os.path.join(new_path, name)
			img = cv2.imread(original, 0)
			equ = cv2.equalizeHist(img)

			if i %100 == 0:
				print("{}, name = {}".format(len(files) - i, name))

			#save new image histogram equ
			cv2.imwrite(output_path, equ)

	return new_path


# compute mean and standard deviation
def mean_std_by(image):
	
	image = Image.open(image)
	pixels = asarray(image)
	
	mean = np.mean(pixels.flatten(), dtype=np.float64)
	std = np.std(pixels.flatten(), dtype=np.float64)

	return mean, std


def compute_mean_std_general(path):
	
	if not os.path.isdir(path):
		raise Exception("Invalid directory...")
	
	means = []
	stds = []
	
	for root, dirs, files in os.walk(path, topdown=False):
		for i, name in enumerate(files, 1):

			original = os.path.join(root, name)
			mean, std = mean_std_by(original)

			means.append(mean)
			stds.append(std)

			if i % 100 == 0:
				print("'{}' computing std, mean...left {}".format(name,
													  len(files) - i))
				
	mean_gral = np.mean(np.asarray(means))
	std_gral = np.std(np.asarray(stds))
	
	return mean_gral, std_gral


def zscore(origin, new_folder, mean, std):
	
	try:
		image = Image.open(origin)
		pixels = asarray(image)
		pixels = (pixels - mean) /  std
		im = Image.fromarray(np.float64(pixels))
		#save it
		im.convert('L').save(new_folder)
		
	except Exception:
		print("An error has ocurred while trying to save a zscore image...")


def zscore_normalization(path, mean, std):
	
	if not os.path.isdir(path):
		raise Exception("Invalid directory")
	
	
	# New Directory
	directory = "ZscoreApplied/"
	new_path = os.path.join(path, directory)
	
	if not os.path.isdir(new_path):
		os.mkdir(new_path)
		
	# compute mean and std for all images
	# 
	#############mean, std = read_images(path)
	print("")
	print("mean gral: {}\nstd gral: {}".format(mean, std))
	print("")
	
	for root, dirs, files in os.walk(path, topdown=False):
		for i, file in enumerate(files):
			
			original = os.path.join(root, file)
			new_folder = os.path.join(new_path, file)
			
			zscore(original, new_folder, mean, std)

			if i%100 == 0:
				print("'{}' comuting z-score... left {}".format(file,
													len(files) - i))

	return new_path
	

def downsampling_images(path, folder, sett):
	
	if not os.path.isdir(path):
		raise Exception("Invalid directory: {}".format(path))
	
	#if not os.path.isdir(final):
	#	raise Exception("Invalid directory {}".format(final))
	
	# New Directory
	# 
	data = "C:/Users/virus/Downloads/CK+/dataset-6emotions/dataset/"
	

	# folder dataset
	if not os.path.isdir(data):
		os.mkdir(data)
		print("creado...1")


	dataset = data + sett
	
	#folder train
	if not os.path.isdir(dataset):
		os.mkdir(dataset)
		print("creado...2")


	directory = folder + "/"
	new_path = os.path.join(dataset, directory)
	

	# folder emotion
	if not os.path.isdir(new_path):
		os.mkdir(new_path)
		print("creado...3")

	for root, dirs, files in os.walk(path, topdown=False):
		for i, file in enumerate(files, 1):
			
			image_original = os.path.join(root, file)

			new_folder = os.path.join(new_path, file)

			if i%100 == 0:
				print("'{}' Appliying downsamplig...left {}".format(file,
														len(files) - i))

			downsampling_apply(image_original, new_folder, size = 32)
			
	return 1

