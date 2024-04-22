import cv2


capture = cv2.VideoCapture(
    'udp://localhost:48697'
)
try:
    while True:
        ret, frame = capture.read()
        if ret:
            cv2.imshow('f', frame)
            cv2.waitKey(1)
except KeyboardInterrupt:
    pass
