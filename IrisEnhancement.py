import cv2
import numpy as np
import matplotlib.pyplot as plt

def enhacement(image):

  

    # Convert to grayscale
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    # Perform histogram equalization
    equalized = cv2.equalizeHist(image)


    return equalized