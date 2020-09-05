import os
import cv2
import numpy as np
from PIL import Image
from numpy import asarray
from scipy import stats

# histogram equalization and z-score normalizacion


def histogram_equ(path):


	if not os.path.isdir(path):
		raise Exception("Invalid directory")
	
	directory = "Histogram/"
	new_path = os.path.join(path, directory)

	for root, dirs, files in os.walk(path, topdown=False):

		if not os.path.isdir(new_path):
			os.mkdir(new_path)

		for i, name in enumerate(files, 1):
			
			original = os.path.join(root, name)
			output_path = os.path.join(new_path, name)
			img = cv2.imread(original, 0)
			equ = cv2.equalizeHist(img)

			
			#res = np.hstack((img,equ)) #stacking images side-by-side

			#save new image histogram equ
			cv2.imwrite(output_path, equ)
			print("{}".format(len(files) - i))



def z_score_norm(image):
	# load image
	image = Image.open(image)
	pixels = asarray(image)
	# confirm pixel range is 0-255
	
	print('dim image: {}'.format(pixels.shape))
	print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))
	# convert from integers to floats
	#pixels = pixels.astype('float32')
	image = stats.zscore(pixels, axis=1, ddof=1)
	image = Image.fromarray(image)

	image.show()
	
if __name__ == '__main__':
	# folder with all images
	dir_image = 'C:/Users/virus/source/repos/FER/CK+/ckplus/'
	# histogram equalization
	histogram_equ(dir_image)


