import numpy as np
import cv2
import imutils

"""
Python 3.6 + OpenCV 3.4.2
IMPLEMENTACION DE DETECTOR SINGLE SHOT DETECTOR SSD para la deteccion de personas
Class_id = 15
Test velocidad: tiempo real...buena ejecucion
"""

inWidth = 300
inHeight = 300
WHRatio = inWidth / float(inHeight)
inScaleFactor = 0.007843
meanVal = 127.5

classNames = ('background',
              'aeroplane', 'bicycle', 'bird', 'boat',
              'botella', 'bus', 'car', 'cat', 'chair',
              'cow', 'diningtable', 'dog', 'horse',
              'motorbike', 'persona', 'pottedplant',
              'sheep', 'sofa', 'train', 'tvmonitor')


net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt','MobileNetSSD_deploy.caffemodel')

cap = cv2.VideoCapture("2.mp4")

#cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame,None,fx=0.5, fy=0.5,
                        interpolation = cv2.INTER_LINEAR)
    frame_resized = cv2.resize(frame,(300,300)) # resize frame for prediction
    frame_copy = frame.copy()
    #cv2.imshow("FRAME ORIGINAL",frame_copy)
    blob = cv2.dnn.blobFromImage(frame_resized, inScaleFactor, (inWidth, inHeight), meanVal)
    net.setInput(blob)
    detections = net.forward()

    cols = frame_resized.shape[1]
    rows = frame_resized.shape[0]

    

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            class_id = int(detections[0, 0, i, 1])
            print("class_id",class_id)

            if class_id == 15 : #or class_id == 5: # personas

                xLeftBottom = int(detections[0, 0, i, 3] * cols)
                yLeftBottom = int(detections[0, 0, i, 4] * rows)
                xRightTop   = int(detections[0, 0, i, 5] * cols)
                yRightTop   = int(detections[0, 0, i, 6] * rows)


                heightFactor = frame.shape[0]/300.0  
                widthFactor = frame.shape[1]/300.0 
                # Scale object detection to frame
                xLeftBottom = int(widthFactor * xLeftBottom) 
                yLeftBottom = int(heightFactor * yLeftBottom)
                xRightTop   = int(widthFactor * xRightTop)
                yRightTop   = int(heightFactor * yRightTop)
                # Draw location of object  
                cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                              (0, 255, 0))

                
                label = classNames[class_id] + ": " + str(confidence)
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                yLeftBottom = max(yLeftBottom, labelSize[1])

                cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                     (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                     (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (xLeftBottom, yLeftBottom),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    cv2.imshow("detections", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
