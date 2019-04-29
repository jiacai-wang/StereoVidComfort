
import numpy as np
import matplotlib
import cv2
import os
from matplotlib import pyplot as plt



img = cv2.imread("./pic_en/2.jpg")

imgL = np.split(img, 2, 1)[0]
imgR = np.split(img, 2, 1)[1]
stereo = cv2.StereoSGBM_create(numDisparities=96, blockSize=11)
disparity = stereo.compute(imgL, imgR)
plt.title("SGBM")
plt.imshow(disparity)
plt.show()
