import ffmpeg
import numpy as np
import matplotlib
import cv2
import os
import sys
from matplotlib import pyplot as plt

cap = cv2.VideoCapture('./vid/zootopia.mkv')
frameCount = cap.get(cv2.CAP_PROP_FRAME_COUNT)
frameRate = cap.get(cv2.CAP_PROP_FPS)

for frameID in range(int(frameRate), int(frameCount), int(frameRate*100)):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frameID)
    isSuccess, img = cap.read()
    if isSuccess:
        cv2.namedWindow("img",cv2.WINDOW_NORMAL);
        cv2.imshow('img', img)
        imgL = np.split(img, 2, 1)[0]
        imgR = np.split(img, 2, 1)[1]
        cv2.waitKey(1)
        stereo = cv2.StereoSGBM_create(numDisparities=96, blockSize=7)
        disparity = stereo.compute(imgL, imgR)
        plt.title("DepthMap")
        plt.imshow(disparity)
        plt.pause(0.1)





# 中文文件名无法识别
# imgDirs = os.listdir("./pic_en")


#def read_frame_as_jpeg(in_filename, frame_num):
#    out, err = (
#        ffmpeg
#        .input(in_filename)
#        .filter('select', 'gte(n,{})'.format(frame_num))
#        .output('pipe:', vframes=1, format='image2', vcodec='mjpeg')
#        .run(capture_stdout=True)
#    )
#    return out

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
# import numpy as np
# import cv2
# from matplotlib import pyplot as plt
# 
# imgL = cv2.imread('tsukuba_l.png',0)
# imgR = cv2.imread('tsukuba_r.png',0)
# 
# stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
# disparity = stereo.compute(imgL,imgR)
# plt.imshow(disparity,'gray')
# plt.show()
