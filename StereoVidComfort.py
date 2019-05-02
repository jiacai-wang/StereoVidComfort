import ffmpeg
import numpy as np
import matplotlib
import cv2
import os
import sys
from matplotlib import pyplot as plt


def openVid():
    fileName = "./vid/zootopia.mkv"
    #fileName = input("video path:")
    while not os.path.isfile(fileName):
        print("file doesn't exist!")
        fileName = input("video path:")
    cap = cv2.VideoCapture(fileName)
    if not cap.isOpened():
        print("Video cannot be opened.")
        sys.exit()
    else:
        return cap


def getFrameCount(cap):
    if not cap.isOpened():
        print("Video cannot be opened.")
        sys.exit()
    else:
        return cap.get(cv2.CAP_PROP_FRAME_COUNT)


def getFrameRate(cap):
    if not cap.isOpened():
        print("Video cannot be opened.")
        sys.exit()
    else:
        return cap.get(cv2.CAP_PROP_FPS)


if __name__ == "__main__":
    cap = openVid()
    frameRate = getFrameRate(cap)
    frameCount = getFrameCount(cap)
    ret, img = cap.read()
    imgR = np.split(img, 2, 1)[1]
    prvs = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(imgR)
    hsv[..., 1] = 255
    cap.set(cv2.CAP_PROP_POS_FRAMES,12000)
    for frameID in range(int(cap.get(cv2.CAP_PROP_POS_FRAMES)), int(frameCount), int(frameRate/2)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frameID)
        isSuccess, img = cap.read()
        if isSuccess:

            imgL = np.split(img, 2, 1)[0]
            imgR = np.split(img, 2, 1)[1]
            next = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang*180/np.pi/2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            cv2.namedWindow("imgR", cv2.WINDOW_NORMAL)
            cv2.imshow('imgR', imgR)
            cv2.namedWindow("MotionVector",cv2.WINDOW_NORMAL)
            cv2.imshow("MotionVector",bgr)
            prvs = next

            stereo = cv2.StereoSGBM_create(numDisparities=96, blockSize=11)
            disparity = stereo.compute(imgL, imgR)
            #cv2.waitKey(1)
            plt.title("DepthMap")
            plt.imshow(disparity)
            plt.pause(0.5)
        else:
            print("video read error")
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
