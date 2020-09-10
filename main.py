import time
import random
import rotations as rt
import data_normalizacion as dn

def delay(n):
	#Delay n seconds for a given number.
	#
	time.sleep(random.uniform(n - 1, n + 1))
	return None

def main():

	
	print("=> Main <=")
	path = 'C:/Users/virus/source/repos/DATASETS/CK+/'
	delay(7)

	# rotate images
	print("\nApplying Rotations...\n")
	new_images_rotated = rt.compute_rotations(path)
	print("\n\n")

	
	delay(5)
	print("\nApplying clipping...\n")
	path_cropped = rt.facial_and_box_landmarcks(new_images_rotated)

	# zscore y despues histogram equalization
	delay(5)
	print("\nComputing the mean and standard deviation for all images...")
	mean_gral, std_gral = dn.compute_mean_std_general(path_cropped)

	delay(5)
	print("\nAppliying the z-score normalizacion...")
	zscore_path = dn.zscore_normalization(path_cropped, mean_gral, std_gral)

	# apply histogram equ
	delay(5)
	print("\nApplying the histogram equalization...")
	final_path = dn.histogram_equ(zscore_path)

	# finally, downsampling
	delay(5)
	print("\nApplying downsampling...")
	exit = dn.downsampling_images(final_path, path)


	if exit:
		print("\n\nFinished...")
	else:
		print("something wrong happened...")


if __name__ == "__main__":

	main()





