########################################## with tracking ##############################################################
# python -m venv venv
# python 3.7.4
#maison : env conda

import cv2 #opencv-contrib-python 4.9.0.80
import  numpy as np
from ultralytics import YOLO # 8.0.145
import time
import torch

print(torch.cuda.is_available())
model = YOLO('YOLO/player_detection on minecraft 04 (gpu).pt')

cap = cv2.VideoCapture('YOLO/the-most-intense-minecraft-pvp-battle.mp4')

# create a dictionary of all trackers in OpenCV that can be used for tracking
OPENCV_OBJECT_TRACKERS = {
	"csrt": cv2.legacy.TrackerCSRT_create, # peu grossir/rapetir est 7/10
	"kcf": cv2.legacy.TrackerKCF_create, # 5/10 
	"boosting": cv2.legacy.TrackerBoosting_create,# 3/10  n'arete pas de tracker
	"mil": cv2.legacy.TrackerMIL_create,# 3/10  n'arete pas de tracker
	"tld": cv2.legacy.TrackerTLD_create,# 4/10  n'arete pas de tracker mais retrouve des chose semblable
	"medianflow": cv2.legacy.TrackerMedianFlow_create,# 4/10  n'arete pas de tracker mais retrouve des chose semblable
	"mosse": cv2.legacy.TrackerMOSSE_create# 4/10  n'arete pas de tracker 
}


trackers = cv2.legacy.MultiTracker_create()
compteur = 5
start_time = time.time()
while True:
    frame = cap.read()[1]

    if frame is None:
        break
    frame = cv2.resize(frame,(1090,600))
    
    if compteur % 7 == 0: # re faire tourner le model toutes les 5 frames
        trackers = cv2.legacy.MultiTracker_create()
        boxes = []
        result = model(frame)
        result = result[0]
        for idx in range(len(result.boxes)):
            box = result.boxes[idx]
            pos = box.xyxy[0]

            x1 = int(pos[0].item())
            y1 = int(pos[1].item())

            x2 = int(pos[2].item())
            y2 = int(pos[3].item())

            w = x2 - x1
            h = y2 - y1

            bound_box = (x1, y1, w, h)
            tracker = OPENCV_OBJECT_TRACKERS['csrt']()
            trackers.add(tracker, frame, bound_box)

            if boxes == ():
                boxes = []
                boxes.append([x1, y1, w, h])
            else:
                try:
                    boxes = boxes.tolist()
                except:
                    pass

                boxes.append([x1, y1, w, h])
    else:
        (success, boxes) = trackers.update(frame)
        
    
        
    for i,box in enumerate(boxes):
        #print("box : ", box)
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('Frame', frame)
    k = cv2.waitKey(30)

    if k == ord("s"): 
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
        roi = cv2.selectROI("Frame", frame, fromCenter=False,
                            showCrosshair=True)
        # create a new object tracker for the bounding box and add it
        # to our multi-object tracker
        tracker = OPENCV_OBJECT_TRACKERS['csrt']()
        trackers.add(tracker, frame, roi)
    
    compteur += 1 
    fps = compteur / (time.time() - start_time)
    print(fps)
    
cap.release()
cv2.destroyAllWindows()