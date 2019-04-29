
import numpy as np
import matplotlib
import cv2
import os
from matplotlib import pyplot as plt



img = cv2.imread("1.png", 0)

imgL = np.split(img, 2, 1)[0]
imgR = np.split(img, 2, 1)[1]
stereo = cv2.StereoBM_create(numDisparities=32, blockSize=25)
disparity = stereo.compute(imgL, imgR)
plt.imshow(disparity)
plt.show()
