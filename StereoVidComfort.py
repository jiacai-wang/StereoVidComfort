import ffmpeg
import numpy as np
import matplotlib
import cv2
import os
import sys
from matplotlib import pyplot as plt

# 中文文件名无法识别
imgDirs = os.listdir("./pic_en")


def read_frame_as_jpeg(in_filename, frame_num):
    out, err = (
        ffmpeg
        .input(in_filename)
        .filter('select', 'gte(n,{})'.format(frame_num))
        .output('pipe:', vframes=1, format='image2', vcodec='mjpeg')
        .run(capture_stdout=True)
    )
    return out

# ffmpeg.input("./vid/venom.mkv")
# ffmpeg.

#img = read_frame_as_jpeg("./vid/venom.mkv", 648)
# print(type(img))
#img = cv2.imdecode(img,0)
# print(img)
# cv2.imshow("img",img)
#imgL = np.split(img, 2, 1)[0]
#imgR = np.split(img, 2, 1)[1]
#stereo = cv2.StereoBM_create(numDisparities=64, blockSize=11)
#disparity = stereo.compute(imgL, imgR)
# plt.imshow(disparity)
# plt.show()


cap = cv2.VideoCapture('./vid/zootopia.mkv')

totalFrame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
for frameID in range(1, int(totalFrame), 1440):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frameID)
    isSuccess, img = cap.read()
    if isSuccess:
        cv2.imshow('img', img)
        imgL = np.split(img, 2, 1)[0]
        imgR = np.split(img, 2, 1)[1]
        stereo = cv2.StereoSGBM_create(numDisparities=96, blockSize=11)
        disparity = stereo.compute(imgL, imgR)
        plt.imshow(disparity)
        plt.show()


#for imgDir in imgDirs:
#    dir = "./pic_en/"+imgDir
#    print(dir)
#    img = cv2.imread(dir)

#    imgL = np.split(img, 2, 1)[0]
#    imgR = np.split(img, 2, 1)[1]
#    print(img.shape)
#    print(imgL.shape)
#    print(imgR.shape)
#    cv2.imshow("img", img)
#    stereo = cv2.StereoSGBM_create(numDisparities=96, blockSize=11)
#    disparity = stereo.compute(imgL, imgR)

#    plt.imshow(disparity)
#    plt.show()
