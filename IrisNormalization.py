import os
import cv2
import numpy as np

def IrisNormalization(boundaries, centers):
    normalized = []

    for boundary_img, center in zip(boundaries, centers):
        # Step 1: Determine the Center and Radius
        center_x, center_y, radius_pupil = center

        # the width of space between inner and outer boundary (iris width)
        iris_radius = 53

        # Step 2: Sample Points in Polar Coordinates (equally spaced interval (360 degrees))
        nsamples = 360
        samples = np.linspace(0, 2*np.pi, nsamples)[:-1]
        polar = np.zeros((iris_radius, nsamples))

        for r in range(iris_radius):
            for theta in samples:
                # convert polar to Cartesian coordinates
                x = int((r + radius_pupil) * np.cos(theta) + center_x)
                y = int((r + radius_pupil) * np.sin(theta) + center_y)

                try:
                    # Step 3: Convert Polar to Cartesian Coordinates
                    polar[r][int((theta * nsamples) / (2 * np.pi))] = boundary_img[y][x]
                except IndexError:

                    pass

        # Step 4: Resize the Normalized Image to a fixed dimension (512x64)
        res = cv2.resize(polar, (512, 64))
        normalized.append(res)

    return normalized  # 64x512 normalized images