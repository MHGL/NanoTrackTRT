# -*- coding:utf-8 -*-
import sys
import cv2
import tracker


video_file = "../samples/test.mp4"

m_tracker = tracker.Tracker()

cap = cv2.VideoCapture(video_file)
if not cap.isOpened():
    print(">>> Open video failed!")
    sys.exit(-1)

cv2.namedWindow("display")

is_tracking = False
while 1:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        is_tracking = False
        continue

    if not is_tracking:
        rect = cv2.selectROI("display", frame, True)
        # print(">>>rect: ", rect)
        m_tracker.init(frame, rect)
        cv2.imshow("display", frame)

        is_tracking = True
        continue

    rect = m_tracker.update(frame)
    x1, y1, w, h = rect
    cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), (0, 255, 0), 1)
    cv2.imshow("display", frame)

    if cv2.waitKey(10) == 27:
        is_tracking = False

cap.release()
