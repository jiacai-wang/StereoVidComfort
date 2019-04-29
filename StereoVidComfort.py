import ffmpeg
import numpy as np
import matplotlib
import cv2
import os
from matplotlib import pyplot as plt


imgDirs = os.listdir("./pic_en")



for imgDir in imgDirs:
    dir = "./pic_en/"+imgDir
    print (dir)
    img = cv2.imread(dir, 0)

    imgL = np.split(img, 2, 1)[0]
    imgR = np.split(img, 2, 1)[1]
    stereo = cv2.StereoBM_create(numDisparities=64, blockSize=11)
    disparity = stereo.compute(imgL, imgR)
    plt.imshow(disparity)
    plt.show()
