"# FER" 
The present project was proposed by Kuan Li et al. 2019 and the CK+ database and JAFFE.


Includes a pre-processing step to training images:

1. Compute the standard deviation and mean of all images (train and test)

2. Face alignment (require a lib https://github.com/davisking/dlib for get facial landmarcks in the step of facealingment)

3. Image cropping

4. Data normalization (z-score and equalization histogram)

5. Downsampling to 32x32

6. Data augmentation (online, random rotation (-2°,2°) and horizontal flipping)



To request CK+: http://www.jeffcohn.net/wp-content/uploads/2020/04/CK-AgreementForm.pdf

JAFFE: https://zenodo.org/record/3451524#.YAmqvohKiUk

Resource: https://link.springer.com/article/10.1007/s00371-019-01627-4
