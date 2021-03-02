import time
import math
import cv2
import numpy as np

confid = 0.5
thresh = 0.5


labelsPath = "coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)

weightsPath = "yolov3.weights"
configPath = "yolov3.cfg"

###### use this for faster processing (caution: slighly lower accuracy) ###########

# weightsPath = "./yolov3-tiny.weights"  ## https://pjreddie.com/media/files/yolov3-tiny.weights
# configPath = "./yolov3-tiny.cfg"       ## https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg


net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
FR=0


vid_path = "vid_short.mp4"
vs = cv2.VideoCapture(vid_path)
# vs = cv2.VideoCapture(0)  ## USe this if you want to use webcam feed
#out = cv2.VideoWriter('outpy4.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1280,930))#720 + 210

while True:

    (grabbed, frame) = vs.read()

    if not grabbed:
        break
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5,
                        interpolation = cv2.INTER_LINEAR)
    (H, W) = frame.shape[:2]
    #print(H,W)
    


    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:

        for detection in output:

            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if LABELS[classID] == "person":

                if confidence > confid:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confid, thresh)

    if len(idxs) > 0:

       
        idf = idxs.flatten()

        for i in idf:
            
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            cen = [int(x + w / 2), int(y + h / 2)]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 100, 100), 2)
            cv2.circle(frame, tuple(cen),1,(0,0,255),4)
            

    cv2.imshow('YOLOv3', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
        
    
    #out.write(result)

#out.release()
vs.release()
cv2.destroyAllWindows()
