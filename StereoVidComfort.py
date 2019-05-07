import ffmpeg
import numpy as np
import matplotlib
import cv2
import os
import sys
from matplotlib import pyplot as plt
from scipy import stats


'''
TODO:
    0:读取视频  √
    1:获取视差  √
    2:获取运动矢量    √
    3:确定舒适度
    4:加舒适度水印
    ...
'''

# 打开视频文件
def openVid():
    fileName = input("video path: ./vid/")
    fileName = "./vid/" + fileName
    while not os.path.isfile(fileName):
        if os.path.isfile(fileName + ".mkv"):
            fileName = fileName + ".mkv"
            break
        print("file doesn't exist!")
        fileName = input("video path: ./vid/")
        fileName = "./vid/" + fileName
    cap = cv2.VideoCapture(fileName)
    if cap.isOpened():
        return cap
    else:
        print("cannot open video.")
        sys.exit()


# 获取视频总帧数
def getFrameCount(cap):
    if cap.isOpened():
        return cap.get(cv2.CAP_PROP_FRAME_COUNT)
    else:
        print("cannot open video.")
        sys.exit()

# 获取帧速率
def getFrameRate(cap):
    if cap.isOpened():
        return cap.get(cv2.CAP_PROP_FPS)
    else:
        print("cannot open video.")
        sys.exit()

# 给出左右画面，计算景深
def getDepthMap(imgL, imgR):
    stereo = cv2.StereoSGBM_create(numDisparities=32, blockSize=3)
    return stereo.compute(imgL, imgR)


# 给出前后两帧，计算帧间运动矢量
def getMotionVector(prvs, next):
    hsv = np.zeros_like(imgR)  # 将运动矢量按hsv显示，以色调h表示运动方向，以明度v表示运动位移
    hsv[..., 1] = 255  # 饱和度置为最高

    # 转为灰度以计算光流
    prvs = cv2.cvtColor(prvs, cv2.COLOR_BGR2GRAY)
    next = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)  # 计算两帧间的光流，即运动矢量的直角坐标表示
    mag, ang = cv2.cartToPolar(
        flow[..., 0], flow[..., 1])  # 运动矢量的直角坐标表示转换为极坐标表示
    hsv[..., 0] = ang*180/np.pi/2  # 角度对应色调
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # 位移量对应明度
    return hsv


if __name__ == "__main__":
    cap = openVid()
    isDemo = int(input("is Demo(0/1)?"))
    frameRate = getFrameRate(cap)
    frameCount = getFrameCount(cap)
    framesCalculated = 0
    isSuccess, img = cap.read()
    if not isSuccess:
        print("video read error.")
        sys.exit()

    # 分割左右画面
    imgL = np.split(img, 2, 1)[0]
    imgR = np.split(img, 2, 1)[1]
    prvs = imgR  # 上一帧的右画面，用于运动矢量计算

    # 每秒取4帧进行计算
    for frameID in range(round(cap.get(cv2.CAP_PROP_POS_FRAMES)), round(frameCount), round(frameRate/4)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frameID)
        isSuccess, img = cap.read()
        if not isSuccess:
            print("video read error.")
            sys.exit()

        # 分割左右画面
        imgL = np.split(img, 2, 1)[0]
        imgR = np.split(img, 2, 1)[1]

        next = imgR  # 当前帧的右画面，用于运动矢量计算
        hsv = getMotionVector(prvs, next)

        # 计算深度图
        disparity = getDepthMap(imgL, imgR)

        framesCalculated += 1

        # 显示计算结果
        print("time: ", round(frameID/frameRate, 2))
        print("AVG depth: ", round(np.mean(disparity), 2))      # 景深的平均值，偏大则意味着负视差，可能不适
        print("AVG motion: ", round(np.mean(hsv[..., 2]), 2))       # 运动矢量大小的平均值，可判断画面大致上是否稳定
        print("Mode depth: ", stats.mode(disparity.reshape(-1))[0][0])      # 景深的众数，由于景深基本不连续，众数意义不大
        print("Mode motion: ", stats.mode(hsv[..., 2].reshape(-1))[0][0])       # 运动矢量大小的众数，一般为0，若较大，说明画面中存在较大面积的快速运动，可能不适
        print("STD depth: ", round(np.std(disparity),2))        # 景深的标准差，若偏大说明景深范围较大，可能不适，但同时也是3D感更强的特征
        print("STD motion: ", round(np.std(hsv[...,2]),2))      # 运动矢量大小的标准差，若偏大说明各部分运动比较不一致，可能需要结合运动矢量的方向作进一步判断，若存在较复杂的运动形式，则可能不适

        print()

        # 当为demo模式时显示当前帧画面、运动矢量图和景深图
        if isDemo:
            # 显示当前帧
            cv2.namedWindow("img", cv2.WINDOW_NORMAL)
            cv2.imshow('img', img)

            # 显示当前帧的运动矢量的hsv表示
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)  # hsv转为rgb用于显示
            cv2.namedWindow("MotionVector", cv2.WINDOW_NORMAL)
            cv2.imshow("MotionVector", bgr)
            # cv2.waitKey(1)
            # 显示当前帧的景深图
            plt.title("DepthMap")
            plt.imshow(disparity)
            # 运动矢量的直方图，方便查看数值
            # plt.title("MotionVector")
            # plt.imshow(hsv[...,2])
            # plt.show()
            plt.pause(0.2)
            input("press to continue")
        prvs = next  # 当前帧覆盖上一帧，继续计算
    print("success")

