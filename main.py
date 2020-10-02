import time
import random
import multiprocessing
import rotations as rt
import data_normalizacion as dn


def delay(n):
	#Delay n seconds for a given number.
	#
	time.sleep(random.uniform(n - 1, n + 1))
	return None

def worker(path, folder):

	print("=> Main <=")
	delay(7)

	# rotate images
	start_time_rotate = time.time()
	print("\n({}) Applying Rotations...\n".format(folder))
	new_images_rotated = rt.compute_rotations(path)
	print("--- %s seconds ---" % (time.time() - start_time_rotate))
	print("\n\n")

	
	delay(5)
	start_time_clipping = time.time()
	print("\n({}) Applying clipping...\n".format(folder))
	path_cropped = rt.facial_and_box_landmarcks(new_images_rotated)
	print("--- %s seconds ---" % (time.time() - start_time_clipping))

	# zscore y despues histogram equalization
	delay(5)
	start_time_mean_std = time.time()
	print("\n({}) Computing the mean and standard deviation for all images...".format(folder))
	###################mean_gral, std_gral = dn.compute_mean_std_general(path_cropped)
	
	mean_gral, std_gral = 102.96767079366812, 9.796655560594619
	print("--- %s seconds ---" % (time.time() - start_time_mean_std))



	delay(5)
	start_time_zscore = time.time()
	print("\n({})Appliying the z-score normalizacion...".format(folder))
	zscore_path = dn.zscore_normalization(path_cropped, mean_gral, std_gral)
	print("--- %s seconds ---" % (time.time() - start_time_zscore))

	# apply histogram equ
	delay(5)
	start_time_histogram = time.time()
	print("\n({})Applying the histogram equalization...".format(folder))
	histogram_path = dn.histogram_equ(zscore_path)
	print("--- %s seconds ---" % (time.time() - start_time_histogram))

	# finally, downsampling
	delay(5)
	
	start_time_downsampling = time.time()
	print("\n({})Applying downsampling...".format(folder))
	exit = dn.downsampling_images(histogram_path, path)
	print("--- %s seconds ---" % (time.time() - start_time_downsampling))

	if exit:
		print("\n\nFinished...")
	else:
		print("something wrong happened...")


if __name__ == "__main__":
	
	# six emotions (anger-LISTO, disgust-LISTO, fear-LISTO, happy-LISTO, sadness, surprise)
	# mean = 102.96767079366812
	# std = 9.796655560594619

	jobs = []
	paths = [('C:/Users/virus/source/repos/DATASETS/train/sadness/','sadness'),
		  ('C:/Users/virus/source/repos/DATASETS/train/surprise/','surprise')]


	for path in paths:
		p = multiprocessing.Process(target=worker, args=(path[0], path[1]))
		jobs.append(p)
		p.start()

	for j in jobs:
		j.join()
		print ('%s.exitcode = %s' % (j.name, j.exitcode))

	# pogramacio dinamica
# https://innerpeace-wu.github.io/2017/03/04/DSaA-Dynamic-Programing/#part3
# https://www.codechef.com/wiki/tutorial-dynamic-programming
# --- 14841.312965154648 seconds --- clipping
# --- 15206.103814125061 seconds --- rotations