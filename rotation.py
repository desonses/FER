import os
import dlib
import numpy as np
from PIL import Image, ImageDraw


def compute_angle(points):

	# as array, much more fast
	points = np.asarray(points)
	
	# select eye right and left
	left = points[0:6] # points 36 to 41
	right = points[6:] # points 42 to 47
	
	# compute numerator
	nsum_1 = np.sum(right[:,1]) # points 42 to 47 => 'y'
	nsum_2 = np.sum(left[:,1])  # points 36 to 41 => 'y'
	
	numerador = nsum_1 - nsum_2
	
	# compute denominator
	dsum_1 = np.sum(right[:,0]) # points 42 to 47 => 'x'
	dsum_2 = np.sum(left[:,0])  # points 36 to 41 => 'x'
	
	denominador = dsum_1 - dsum_2
	
	# compute angle = arctan(x)
	angle = np.degrees(np.arctan(numerador / denominador))
	
	return angle

def compute_distance(points):
	
	# as array, much more fast
	points = np.asarray(points)
	
	# select eye right and left
	left = points[0:6] # points(x,y), 36 to 41	
	right = points[6:] # points(x,y), 42 to 47
	
	sum_x = np.sum(right[:,0])
	sum_y = np.sum(left[:,0])
	
	d = (sum_x - sum_y) / 6
	
	return d


def rotation_by(image, angle, new_folder):

	# Create an Image object from an Image
	colorImage  = Image.open(image)
	
	# Rotate it by n-degrees
	rotated = colorImage.rotate(angle)

	rotated.convert('L').save(new_folder)
	# Display the Image rotated
	# ########rotated.show()


def crop_image(points):

	im = im.crop((165, 168, 363, 378))


def display_image(image, points, box):
	
	# convert(), for draw color points in a image in grayscale
	im = Image.open(image).convert('RGB')
	
	draw = ImageDraw.Draw(im)
	
	draw.point(points, fill='red')

	###### draw box
	a = box[0]
	b = box[1]
	c = box[2]
	d = box[3]

	draw.line((a, b), fill ='blue', width = 0)
	draw.line((b, c), fill ='blue', width = 0)
	draw.line((c, d), fill ='blue', width = 0)
	draw.line((d, a), fill ='blue', width = 0)
	
	
	# display
	im.show()



def get_box(points):

	a = points[0]
	b = points[1]
	c = points[2]
	d = points[3]

	square = ((a[0], b[1]), (c[0], b[1]), (c[0], d[1]), (a[0], d[1]))

	return square


def compute_landmarcks(image):
	
	predictor_path = 'C:/Users/virus/Downloads/prueba/shape_predictor_68_face_landmarks.dat'


	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(predictor_path)

	img = dlib.load_rgb_image(image)
	dets = detector(img, 1)

	points = []

	for k, d in enumerate(dets):

		#print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
		#	k, d.left(), d.top(), d.right(), d.bottom()))

		# Get the landmarks/parts for the face in box d.
		landmarcks = predictor(img, d)

		for i in range(68):

			# points 0 to 67
			x = int(landmarcks.part(i).x)
			y = int(landmarcks.part(i).y)
			points.append((x, y))
	

	eyes = points[36:48]

	left = np.asarray(eyes[0:6]) # points(x,y), 36 to 41	
	right = np.asarray(eyes[6:]) # points(x,y), 42 to 47
	
	centroid = computeCentroid(np.asarray(eyes))

	d = compute_distance(eyes) # compute 2nd point to the distance of 0.6

	eyes.append(centroid)
	new_point = (centroid[0], centroid[1] - 60)
	eyes.append(new_point)


	eyes.append(points[0])  # 1
	eyes.append(points[8])  # 9
	eyes.append(points[16]) # 17

	centroid_left = computeCentroid(left)   # (x, y)
	centroid_right = computeCentroid(right) # (x, y)

	eyes.append(centroid_left)
	eyes.append(centroid_right)

	m = (centroid_right[1] - centroid_left[1]) / (centroid_right[0] - centroid_left[0])
	#print("pendiente: {}".format(m))

	#########################
	
	
	rectangle = (points[0], points[8], points[16], new_point)
	box = get_box(rectangle)


	# display image with eyes points
	display_image(image, eyes, box)

	#rotation
	##########rotation_by(image, angle)
	##########return angle, distance
	
def computeCentroid(points):

	xs = int(np.sum([p[0] for p in points]) / len(points))
	ys = int(np.sum([p[1] for p in points]) / len(points))

	centroid = (xs, ys)
	#print("==>", centroid)

	return centroid


def compute_rotations(path):
	
	
	if not os.path.isdir(path):
		raise Exception("Invalid directory")

	# New Directory  
	directory = "Rotations/"
	new_path = os.path.join(path, directory)

	# create new folder
	if not os.path.isdir(new_path):
		os.mkdir(new_path)
	
	for root, dirs, files in os.walk(path, topdown=False):
		for i, file in enumerate(files):

			image = os.path.join(root, file)
			new_folder = os.path.join(new_path, file)

			# get the facial landmarcks and compute the angle
			angle, distance = compute_landmarcks(image)
			print("name: {}\nangle = {}\ndistance = {}".format(file,
													  angle,
													  distance))

			# make rotation and save it in the new directory
			rotation_by(image, angle, new_folder)

			#print("original:   {}".format(original))
			#print("new_folder: {}".format(new_folder))
			print("\n")





if __name__ == '__main__':

	image = 'C:/Users/virus/source/repos/DATASETS/PRUEBA/landmarck/Rotations/rotate2.png'
	compute_landmarcks(image)


	##########path = 'C:/Users/virus/source/repos/DATASETS/PRUEBA/landmarck/'

	##########compute_rotations(path)



