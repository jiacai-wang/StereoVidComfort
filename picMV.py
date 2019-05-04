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