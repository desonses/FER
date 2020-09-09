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


	#return rotated
	rotated.convert('L').save(new_folder)
	# Display the Image rotated
	# ########rotated.show()


def crop_image(points):

	im = im.crop((165, 168, 363, 378))


def display_image(image, points, box, centroids = None, new_folder = None):
	
	# convert(), for draw color points in a image in grayscale
	im = Image.open(image).convert('RGB')
	
	draw = ImageDraw.Draw(im)
	
	draw.point(points, fill='red')

	###### draw box
	a = box[0]
	b = box[1]
	c = box[2]
	d = box[3]

	A = points[-2]
	B = points[-1]

	draw.line((a, b), fill ='blue', width = 0)
	draw.line((b, c), fill ='blue', width = 0)
	draw.line((c, d), fill ='blue', width = 0)
	draw.line((d, a), fill ='blue', width = 0)

	draw.line((A, B), fill ='green', width = 0)
	
	if centroids is not None:
		left = centroids[0]
		right = centroids[1]

		draw.line((left, right), fill ='green', width = 0)

		im.show()

	if new_folder is not None:
		im.convert('RGB').save(new_folder)



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

		# Get the landmarks/parts for the face in box d.
		landmarcks = predictor(img, d)

		for i in range(68):

			# points 0 to 67
			x = int(landmarcks.part(i).x)
			y = int(landmarcks.part(i).y)
			points.append((x, y))
	

	#################eyes = points[36:48]

	################angle = compute_angle(eyes)
	################distance = compute_distance(eyes) # compute 2nd point to the distance of 0.6

	#################left = np.asarray(eyes[0:6]) # points(x,y), 36 to 41	
	#################right = np.asarray(eyes[6:]) # points(x,y), 42 to 47
	
	#centroid = computeCentroid(np.asarray(eyes))

	

	################eyes.append(centroid)
	################new_point = (centroid[0], centroid[1] - 60)
	################eyes.append(new_point)


	################eyes.append(points[0])  # 1
	################eyes.append(points[8])  # 9
	################eyes.append(points[16]) # 17

	################centroid_left = computeCentroid(left)   # (x, y)
	################centroid_right = computeCentroid(right) # (x, y)

	################eyes.append(centroid_left)
	################eyes.append(centroid_right)

	################m = (centroid_right[1] - centroid_left[1]) / (centroid_right[0] - centroid_left[0])
	#############print("slope: {}\n".format(m))

	#########################
	
	
	################rectangle = (points[0], points[8], points[16], new_point)
	################box = get_box(rectangle)


	# display image with eyes points
	###########display_image(image, eyes, box)

	#rotation
	##########rotation_by(image, angle)

	return points

def centroid_of(eyes):

	left = np.asarray(eyes[0:6]) # points(x,y), 36 to 41	
	right = np.asarray(eyes[6:]) # points(x,y), 42 to 47

	centroid_left = computeCentroid(left)   # (x, y)
	centroid_right = computeCentroid(right) # (x, y)

	centroids = (centroid_left, centroid_right)

	return centroids

def compute_slope(centroids):

	centroid_left, centroid_right = centroids[0], centroids[1]
	m = (centroid_right[1] - centroid_left[1]) / (centroid_right[0] - centroid_left[0])

	return m

	
def computeCentroid(points):

	xs = int(np.sum([p[0] for p in points]) / len(points))
	ys = int(np.sum([p[1] for p in points]) / len(points))

	centroid = (xs, ys)

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

			# rotation 
			landmarcks = compute_landmarcks(image)
			eyes = landmarcks[36:48]
			# compute the angle between eyes
			angle = compute_angle(eyes) 

			# get the centroids of eyes, centroids = pupile
			centroids = centroid_of(eyes)

			# compute the slope between centrois, 
			slope = compute_slope(centroids)
			print("{}, '{}', angle = {}, m = {}\n".format(len(files) - i,
												 file,
												 angle,
												 slope))

			# display images
			#im = Image.open(image).convert('RGB')
			#draw = ImageDraw.Draw(im)
			#eyes.append(centroids[0]) # centroid_left, 
			#eyes.append(centroids[1]) # centroid_right

			#draw.point(eyes, fill='red')
			#im.show()
			
			rotation_by(image, angle, new_folder)


	print("finish...")

	
			# get the facial landmarcks and compute the angle
			#print("image ====> {}".format(image))
			################distance = compute_landmarcks(image)["distance"] 
			################box = compute_landmarcks(image)["box"] 
			################eyes = compute_landmarcks(image)["eyes"] 
			################landmarcks = compute_landmarcks(image)["landmarcks"] 

			#print("angle = {}\ndistance = {}\nbox = {}\neyes = {}\nlandmarcks = {}\n\n".format(angle,
			#																	  distance,
			#																	  box,
			#																	  eyes,
			#																	  landmarcks[0:5]))

			################display_image(image, eyes, box)
			# make rotation and save it in the new directory
			

			#print("original:   {}".format(original))
			#print("new_folder: {}".format(new_folder))
			########print("\n")
	return new_path


def facial_and_box_landarmarcks(path):

	if not os.path.isdir(path):
		raise Exception("Invalid directory")

	# New Directory  
	directory = "LandMarcks/"
	new_path = os.path.join(path, directory)

	# create new folder
	if not os.path.isdir(new_path):
		os.mkdir(new_path)


	for root, dirs, files in os.walk(path, topdown=False):
		for i, file in enumerate(files):

			image = os.path.join(root, file)
			new_folder = os.path.join(new_path, file)

			landmarcks = compute_landmarcks(image)

			eyes = landmarcks[36:48]
			# compute the angle between eyes
			angle = compute_angle(eyes) 

			# get the centroids of eyes, centroids = pupile
			centroids = centroid_of(eyes)

			# compute the slope between centrois, 
			slope = compute_slope(centroids)
			print("{}, '{}', angle = {}, m = {}\n".format(len(files) - i,
												 file,
												 angle,
												 slope))
			
			# centroid between eyes
			centroid = computeCentroid(np.asarray(eyes))
			eyes.append(centroid)

			# new point
			new_point = (centroid[0], centroid[1] - 60)
			eyes.append(new_point)

			rectangle = (landmarcks[0],
				landmarcks[8],
				landmarcks[16],
				new_point)

			# box that define the face
			box = get_box(rectangle)


			display_image(image, eyes, box, centroids, new_folder)






if __name__ == '__main__':

	#image = 'C:/Users/virus/source/repos/DATASETS/PRUEBA/landmarck/Rotations/rotate2.png'
	#compute_landmarcks(image)

	# rotate images
	path = 'C:/Users/virus/source/repos/DATASETS/PRUEBA/landmarck/'
	new_images_rotated = compute_rotations(path)

	#################################################################
	#################################################################

	# compute the box area for crop image
	#path = 'C:/Users/virus/source/repos/DATASETS/PRUEBA/landmarck/Rotations/'
	print("new directory = {}".format(new_images_rotated))

	facial_and_box_landarmarcks(new_images_rotated)


