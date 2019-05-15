import cv2 as cv
import numpy as np


cap = cv.VideoCapture("./vid/zootopia.mkv")
ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255
print("jump ahead")
cap.set(cv.CAP_PROP_POS_FRAMES, 4)
print("jump done")

while(1):
    for i in range(1, 12):
        cap.read()
    ret, frame2 = cap.read()
    next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(
        prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    cv.namedWindow("MotionVector", cv.WINDOW_NORMAL)
    cv.namedWindow("frame", cv.WINDOW_NORMAL)
    cv.imshow('MotionVector', bgr)
    cv.imshow("frame", frame2)
    cv.waitKey(1)
    #k = cv.waitKey(0.1) & 0xff
    # if k == 27:
    #    break
    # elif k == ord('s'):
    #    cv.imwrite('opticalfb.png',frame2)
    #    cv.imwrite('opticalhsv.png',bgr)
    prvs = next
cap.release()
cv.destroyAllWindows()
