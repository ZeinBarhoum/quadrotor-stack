import cv2

import logging

logging.basicConfig(level=logging.DEBUG)
capture = cv2.VideoCapture('tcp://192.168.1.1:5555')
while True:
    ret, im = capture.read()
    cv2.imshow('im', im)
    if cv2.waitKey(30) == ord('q'):
        break
