# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 11:50:02 2020

@author: GANGASAGAR
"""

import packages.social_distancing_config as config
from packages.object_detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import imutils
import cv2
import os

#load the coco class labels our YOLO model was trained on
labelsPath = os.path.sep.join([config.MODEL_PATH,"coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

#derive the paths to the YOLO weigts and model configuration
weightsPath = os.path.sep.join([config.MODEL_PATH,"yolov3.weights"])
configPath = os.path.sep.join([config.MODEL_PATH,"yolov3.cfg"])

#load our YOLOobject detector trained on coco dataset
print("[INFO] Loading YOLO from disk ....")
net=cv2.dnn.readNetFromDarknet( configPath , weightsPath )

if config.USE_GPU:
    
    #set cuda as preferable backend and target
    print("[INFO] setting preferable backend and target to CUDA....")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
#determine only the output layer names that we need from YOLO

ln = net.getLayerNames()
ln = [ln[ i[0]-1 ] for i in net.getUnconnectedOutLayers()]

print("[INFO] accessing video stream....")

vs = cv2.VideoCapture("C:/Users/ruchi/Downloads/song.mp4")
global writer
writer = None


#Detecting Person in frame

while True:
    
    (grabbed , frame) = vs.read()
    
    if not grabbed:
        break
    
    frame = imutils.resize(frame , width = 700) 
    results = detect_people(frame,net,ln,personIdx = LABELS.index('person') )
    
    violate = set()
    
    if len(results) >= 2:
        centroids = np.array([r[2] for r in results])
        D = dist.cdist(centroids,centroids,metric='euclidean')
        
        for i in range(0,D.shape[0]):
            for j in range(i+1,D.shape[1]):
                
                if D[ i,j ] < config.MIN_DISTANCE:
                    violate.add(i)
                    violate.add(j)
                    
                    
    for (i,(prob,bbox,centroid)) in enumerate(results):
        
        (startX,startY,endX,endY)=bbox
        (cx,cy) = centroid
        
        color = (0,255,0)
        
        if i in violate:
            color = (0,0,255)
            
        cv2.rectangle(frame,(startX,startY),(endX,endY),color,2)
        cv2.circle(frame,(cx,cy),5,color,1)
        
    text="Social Distncing Violations: {} ".format(len(violate))
    cv2.putText(frame,text,(10,frame.shape[0] - 25 ),
                cv2.FONT_HERSHEY_SIMPLEX,0.85,(0,0,255),3)
    
    cv2.imshow("frame",frame)
    key=cv2.waitKey(1) & 0xFF
    
    if key==ord('q'):
        break
    
    '''if r'social-distance-detector2' != '' and writer is None:
        fourcc=cv2.ViedoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(r"output.mp4",fourcc,25,
                                 (frame.shape[1],frame.shape[0]),True)
        
        if writer is not None:
            writer.write(frame)'''
            
cv2.destroyAllWindows()