import cv2
import numpy as np
from scipy import ndimage
# import matplotlib.pyplot as plt


def feature_extraction(image):

   # Load an iris image (you should replace this with your image loading code)
    # iris_image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    iris_image =image

    # Define Gabor filter parameters
    kernels = []
    for theta in range(192):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for frequency in (0.1, 0.2):
                kernel = np.real(ndimage.gaussian_filter(iris_image, sigma=(sigma, sigma), order=0) * np.exp(1j * (frequency * np.pi * iris_image + theta)))
                kernels.append(kernel)

    # Extract features using Gabor filters
    feature_vectors = []
    for kernel in kernels:
        filtered_iris = cv2.filter2D(iris_image, cv2.CV_64F, kernel)
        feature_vectors.append(filtered_iris.mean())
        feature_vectors.append(filtered_iris.std())

    return feature_vectors
        