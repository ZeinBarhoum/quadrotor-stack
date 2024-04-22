from pyardrone.video import VideoClient
import cv2

import logging

logging.basicConfig(level=logging.DEBUG)

video_client = VideoClient(host="192.168.1.1", video_port=5555)
video_client.connect()
video_client.video_ready.wait()

try:
    while True:
        cv2.imshow('im', video_client.frame)
        if cv2.waitKey(30) == ord('q'):
            break
finally:
    video_client.close()
