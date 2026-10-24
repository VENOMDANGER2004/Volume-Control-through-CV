import cv2
import numpy as np
import time

#####
wCam,hCam = 640,480

cap = cv2.VideoCapture(1)
cap.set(3,wCam)
cap.set(4,hCam)
pTime = 0

while True:
    success, img = cap.read()

    cTime = time.time()
    fps= 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,f'FPS: {int(fps)}',(40,60),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    cv2.imshow('img',img)

    cv2.waitKey(1)







