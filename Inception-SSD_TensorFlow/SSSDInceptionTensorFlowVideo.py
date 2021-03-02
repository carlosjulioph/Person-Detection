# Object Detection using SSD Inception arquitecture trained on COCO dataset
import cv2
import sys
import numpy as np
import time
import imutils


FROZEN_GRAPH = "ssd_inception_v2_coco.pb"
PB_TXT = "ssd_inception_v2_coco.pbtxt"
SIZE = 300

from coco_labels import LABEL_MAP

def run(img):
    cvNet = cv2.dnn.readNetFromTensorflow(FROZEN_GRAPH, PB_TXT)

    rows = img.shape[0]
    cols = img.shape[1]
    cvNet.setInput(cv2.dnn.blobFromImage(img, 1.0/127.5, (SIZE, SIZE), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    cvOut = cvNet.forward()

    for detection in cvOut[0,0,:,:]:
        score = float(detection[2])
        if score > 0.3:
            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows

            if LABEL_MAP[int(detection[1])] == "persona" :#or LABEL_MAP[int(detection[1])] == "truck":
            
                cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)
                cv2.putText(img, LABEL_MAP[int(detection[1])], (int(left), int(top)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

                #print("detection",LABEL_MAP[int(detection[1])])

    return img

cap = cv2.VideoCapture("2.mp4")
#cap = cv2.VideoCapture(0)
out1 = cv2.VideoWriter('outpy4.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (960,540))
while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame,None,fx=0.5, fy=0.5,
                        interpolation = cv2.INTER_LINEAR)
        print(frame.shape)
        image = run(frame)
        cv2.imshow("detections", image)
            
        out1.write(image)
        #Capturamos teclado
        tecla = cv2.waitKey(1) & 0xFF
        #Salimos si la tecla presionada es ESC
        if tecla == 27:
                 break
#Liberamos objeto                                                  
cap.release()
out1.release()
#Destruimos ventanas
cv2.destroyAllWindows()

