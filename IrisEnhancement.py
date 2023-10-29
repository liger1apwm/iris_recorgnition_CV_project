import cv2
import numpy as np
import matplotlib.pyplot as plt

def enhacement(image):

    # # Load image
    # image = cv2.imread('./iris_normalized_test.png') 

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # # Get histogram
    # histogram = cv2.calcHist([gray],[0],None,[256],[0,256])

    # Perform histogram equalization
    equalized = cv2.equalizeHist(gray)

    # # Get equalized histogram 
    # hist_equalized = cv2.calcHist([equalized],[0],None,[256],[0,256]) 

    # # Plot histograms
    # plt.plot(histogram)
    # plt.plot(hist_equalized)
    # plt.xlim([0,256])
    # plt.show()

    # # Display images
    # cv2.imshow('Original', image)
    # cv2.imshow('Histogram Equalized', equalized)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return equalized