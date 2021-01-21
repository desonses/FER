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

	if "test" in path:
		sett = "test/"
	else:
		sett = "train/"


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



	# apply histogram equ
	delay(5)
	start_time_histogram = time.time()
	print("\n({})Applying the histogram equalization...".format(folder))
	histogram_path = dn.histogram_equ(path_cropped)
	print("--- %s seconds ---" % (time.time() - start_time_histogram))


	# computing mean and std
	####################delay(5)
	####################start_time_mean_std = time.time()
	####################print("\n({}) Computing the mean and standard deviation for all images...".format(folder))
	###################mean_gral, std_gral = dn.compute_mean_std_general(path_cropped)
	
	#mean_gral, std_gral = 101.92048226683615, 9.482495051361877 # for CK+
	mean_gral, std_gral = 111.38843198682441, 8.282619906346753 # for JAFFE 6-expressions
	

	#print("--- %s seconds ---" % (time.time() - start_time_mean_std))


	# apply z-score
	delay(5)
	start_time_zscore = time.time()
	print("\n({})Appliying the z-score normalizacion...".format(folder))
	zscore_path = dn.zscore_normalization(histogram_path, mean_gral, std_gral)
	print("--- %s seconds ---" % (time.time() - start_time_zscore))

	

	# finally, downsampling
	delay(5)
	
	start_time_downsampling = time.time()
	print("\n({})Applying downsampling...".format(folder))
	#final ="C:/Users/virus/Downloads/CK+/dataset-6emotions/test/test/" 
	exit = dn.downsampling_images(zscore_path, folder, sett) # final or path=same folder
	
	print("--- %s seconds ---" % (time.time() - start_time_downsampling))

	if exit:
		print("\n\nFinished...")
	else:
		print("something wrong happened...")


if __name__ == "__main__":
	
	# six emotions (anger-LISTO, disgust-LISTO, fear-LISTO, happy-LISTO, sadness, surprise)
	# mean=102.84447893612776
	# std=9.715599506559709

	jobs = []
	paths = [
		('C:/Users/virus/Downloads/CK+/JAFFE_test/anger/','anger'),
		('C:/Users/virus/Downloads/CK+/JAFFE_test/disgust/','disgust'),
		('C:/Users/virus/Downloads/CK+/JAFFE_test/fear/','fear'),
		('C:/Users/virus/Downloads/CK+/JAFFE_test/happy/','happy'),
		('C:/Users/virus/Downloads/CK+/JAFFE_test/sadness/','sadness'),
		('C:/Users/virus/Downloads/CK+/JAFFE_test/surprise/','surprise')
		]


	for path in paths:
		p = multiprocessing.Process(target=worker, args=(path[0], path[1]))
		jobs.append(p)
		p.start()

	for j in jobs:
		j.join()
		print ('%s.exitcode = %s' % (j.name, j.exitcode))

	# computing mean and std
	#path = "C:/Users/virus/Downloads/CK+/JAFFE_test/"
	#delay(5)
	#start_time_mean_std = time.time()
	#print("\n({}) Computing the mean and standard deviation for all images...".format(folder))
	#mean_gral, std_gral = dn.compute_mean_std_general(path)
	#print("mean_gral={}, std_gral={}".format(mean_gral, std_gral))

	#dir = "C:/Users/virus/Downloads/CK+/NEW_dataset-6emotions/train/surprise/"

	
	#path_cropped = rt.facial_and_box_landmarcks(dir)
	





	# pogramacio dinamica
# https://innerpeace-wu.github.io/2017/03/04/DSaA-Dynamic-Programing/#part3
# https://www.codechef.com/wiki/tutorial-dynamic-programming
# --- 14841.312965154648 seconds --- clipping
# --- 15206.103814125061 seconds --- rotations