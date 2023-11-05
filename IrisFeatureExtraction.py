import cv2
import numpy as np
from scipy import ndimage
# import matplotlib.pyplot as plt

# DIFFERENT METHOD FUNCTION 
# def feature_extraction(image):

#    # Load an iris image (you should replace this with your image loading code)
#     # iris_image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
#     iris_image =image

#     # Define Gabor filter parameters
#     kernels = []
#     for theta in range(192):
#         theta = theta / 4. * np.pi
#         for sigma in (1, 3):
#             for frequency in (0.1, 0.2):
#                 kernel = np.real(ndimage.gaussian_filter(iris_image, sigma=(sigma, sigma), order=0) * np.exp(1j * (frequency * np.pi * iris_image + theta)))
#                 kernels.append(kernel)

#     # Extract features using Gabor filters
#     feature_vectors = []
#     for kernel in kernels:
#         filtered_iris = cv2.filter2D(iris_image, cv2.CV_64F, kernel)
#         feature_vectors.append(filtered_iris.mean())
#         feature_vectors.append(filtered_iris.std())

#     return feature_vectors

def create_filter(dx,dy,crop_amount):

    # Filter 1 parameters
    f = 1/dy
    # sig_x = dx
    # sig_y = dy

    x, y = np.mgrid[-7:8, -7:8]
    # x, y = np.mgrid[0:crop_amount, 0:512]

    filter = (1/(2*np.pi*dx*dy))*np.exp(-(x**2 / (2*dx**2) + y**2 / (2*dy**2))) * np.cos(2 * np.pi * f * (x**2 + y**2)**0.5)
    
    return filter

def split_blocks(image, block_size=(8,8)):

  rows = image.shape[0]//block_size[0]
  cols = image.shape[1]//block_size[1]

  blocks = []
  for row in range(rows):
    for col in range(cols):
      block = image[row*block_size[0]:(row+1)*block_size[0], 
                    col*block_size[1]:(col+1)*block_size[1]]
      blocks.append(block)

  return blocks

def feature_extraction(image,crop_amount):

    feature_vectors = []

    

    # Define filters
    filter1 = create_filter(3,1.5,crop_amount)
    filter2 = create_filter(4.5,1.5,crop_amount)

    filtered1 = cv2.filter2D(image, -1, filter1)
    filtered2 = cv2.filter2D(image, -1, filter2)

    blocks_image1 = split_blocks(filtered1)
    blocks_image2 = split_blocks(filtered2)

    for block in blocks_image1:
        block_mean = np.mean(block,axis=(0,1))
        absolute_deviations = np.abs(block - block_mean)
        aad = np.mean(absolute_deviations, axis=(0, 1))

        feature_vectors.append(block_mean)
        feature_vectors.append(aad)

    for block in blocks_image2:
        block_mean = np.mean(block,axis=(0,1))
        absolute_deviations = np.abs(block - block_mean)
        aad = np.mean(absolute_deviations, axis=(0, 1))

        feature_vectors.append(block_mean)
        feature_vectors.append(aad)


    # print(len(blocks_image1))


    return feature_vectors

    
