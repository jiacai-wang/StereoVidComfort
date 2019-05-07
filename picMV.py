import cv2 as cv
import numpy as np

img = cv.imread("./pic/2.jpg")
imgL = np.split(img, 2, 1)[0]
imgR = np.split(img, 2, 1)[1]

prvs = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(imgL)
hsv[..., 1] = 255

next = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
hsv[..., 0] = ang*180/np.pi/2
hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
cv.namedWindow("MotionVector", cv.WINDOW_NORMAL)
cv.namedWindow("frame", cv.WINDOW_NORMAL)
cv.imshow('MotionVector', bgr)
cv.imshow("frame", img)
cv.waitKey(100)


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
