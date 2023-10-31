# import cv2 
# import numpy as np 

s1 = "hello"
s2 = "world"

s1 = s1[::-1]
s2= s2[::-1]
s = ""
i = 0 
while i < len(s1):

    s = s + s1[i]
    s = s + s2[i]
    i+=1

print(s)