import cv2 
import numpy as np 

theta = 0.001 

x = np.array([-3.3, 0.1, -1.1, 2.7, 2.0,-0.4])
y = np.array([-2.6,-0.2,-1.5,1.5,1.9,-0.3])

sum = sum(x**2+ 2*theta*x*y+y**2)

print(sum)