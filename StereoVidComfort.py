import ffmpeg
import numpy as np
import matplotlib
import cv2
import os
import sys
from matplotlib import pyplot as plt


'''
TODO:
    0:读取视频  √
    1:获取视差  √
    2:获取运动矢量    √
    3:确定舒适度
    4:加舒适度水印
    ...
'''

def openVid():
    fileName = input("video path:")
    while not os.path.isfile(fileName):
        print("file doesn't exist!")
        fileName = input("video path:")
    cap = cv2.VideoCapture(fileName)
    if cap.isOpened():
        return cap
    else:
        print("cannot open video.")
        sys.exit()
        


def getFrameCount(cap):
    if cap.isOpened():
        return cap.get(cv2.CAP_PROP_FRAME_COUNT)
    else:
        print("cannot open video.")
        sys.exit()


def getFrameRate(cap):
    if cap.isOpened():
        return cap.get(cv2.CAP_PROP_FPS)
    else:
        print("cannot open video.")
        sys.exit()


if __name__ == "__main__":
    cap = openVid()
    isDemo = int(input("is Demo(0/1)?"))
    frameRate = getFrameRate(cap)
    frameCount = getFrameCount(cap)

    isSuccess, img = cap.read()
    if not isSuccess:
        print("video read error.")
        sys.exit()

    #分割左右画面
    imgL = np.split(img, 2, 1)[0]
    imgR = np.split(img, 2, 1)[1]
    prvs = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)       #前一帧的右画面灰度，用于运动矢量计算
    hsv = np.zeros_like(imgR)       #将运动矢量按hsv显示，以色调h表示运动方向，以明度v表示运动位移
    hsv[..., 1] = 255       #饱和度置为最高

    #每秒取4帧进行计算
    for frameID in range(round(cap.get(cv2.CAP_PROP_POS_FRAMES)), round(frameCount), round(frameRate/4)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frameID)
        isSuccess, img = cap.read()
        if not isSuccess:
            print("video read error.")
            sys.exit()

        #分割左右画面
        imgL = np.split(img, 2, 1)[0]
        imgR = np.split(img, 2, 1)[1]

        next = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)       #当前帧的右画面灰度，用于运动矢量计算
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)     #计算两帧间的光流
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])      #运动矢量的直角坐标表示转换为极坐标表示
        hsv[..., 0] = ang*180/np.pi/2       #角度对应色调
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)     #位移量对应明度

        #计算深度图
        stereo = cv2.StereoSGBM_create(numDisparities=64, blockSize=3)
        disparity = stereo.compute(imgL, imgR)
        print("time: ", round(frameID/frameRate,2))
        print("AVG depth: ",round(np.mean(disparity),2))
        print("AVG motion: ",round(np.mean(hsv[...,2]),2))
        print()
        
        #当为demo模式时显示当前帧画面、运动矢量图和景深图
        if isDemo:
            #显示当前帧
            cv2.namedWindow("img", cv2.WINDOW_NORMAL)
            cv2.imshow('img', img)

            #显示当前帧的运动矢量的hsv表示
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)      #hsv转为rgb用于显示
            cv2.namedWindow("MotionVector",cv2.WINDOW_NORMAL)
            cv2.imshow("MotionVector",bgr)
            
            #显示当前帧的景深图
            plt.title("DepthMap")
            plt.imshow(disparity)
            plt.pause(0.5)

        prvs = next     #当前帧覆盖上一帧，继续计算
    print("success")






# ffmpeg.input("./vid/avatar.mkv")

# Motion Vector
#cap = cv2.VideoCapture('./vid/zootopia.mkv')
# for i in range(1,10000):
#    cap.read()
# params for ShiTomasi corner detection
# feature_params = dict( maxCorners = 100,
#                       qualityLevel = 0.3,
#                       minDistance = 7,
#                       blockSize = 7 )
# Parameters for lucas kanade optical flow
# lk_params = dict( winSize  = (15,15),
#                  maxLevel = 2,
#                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
#color = np.random.randint(0,255,(100,3))
# Take first frame and find corners in it
#ret, old_frame = cap.read()
#old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
#p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# Create a mask image for drawing purposes
#mask = np.zeros_like(old_frame)
# while(1):
#    ret,frame = cap.read()
#    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#    # calculate optical flow
#    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
#    # Select good points
#    good_new = p1[st==1]
#    good_old = p0[st==1]
#    # draw the tracks
#    for i,(new,old) in enumerate(zip(good_new,good_old)):
#        a,b = new.ravel()
#        c,d = old.ravel()
#        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
#        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
#    img = cv2.add(frame,mask)
#    cv2.imshow('frame',img)
#    k = cv2.waitKey(30) & 0xff
#    if k == 27:
#        break
#    # Now update the previous frame and previous points
#    old_gray = frame_gray.copy()
#    p0 = good_new.reshape(-1,1,2)
# cv2.destroyAllWindows()
# cap.release()


# 中文文件名无法识别
# imgDirs = os.listdir("./pic_en")


# def read_frame_as_jpeg(in_filename, frame_num):
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


# for imgDir in imgDirs:
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
