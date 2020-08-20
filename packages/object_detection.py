# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 11:45:42 2020

@author: GANGASAGAR
"""

from packages.social_distancing_config import MIN_CONF
from packages.social_distancing_config import NMS_THRESH

import numpy as np
import cv2




# It takes data from social distancing for people dtection in video or data we provide
# It preprocess the picture and gives back 
# result to the model
#extracts the dimension of the frame and 
#initialise the list result
#after detecting it gives back the 
# persoms data to model

    
    #NMS-used to smoothen data
 
def detect_people(frames,net,ln,personIdx=0):
    
    
    (H,W)=frames.shape[:2]
    
    #This results holds data from yolo algorithm

    results=[]    
    
    blob=cv2.dnn.blobFromImage(frames,1/255.0,(416,416),
                               swapRB=True,crop=False)
    
    net.setInput(blob)
    layerOutputs=net.forward(ln)
    
    #Initialising our list of bounding boxes,centroid,confidence respectively
    
    boxes = []
    centroids = []
    confidences = []
    
    
    for output in layerOutputs:
        for detection in output:
            
            # detection - confidence , x,y,h,w,pr(1),...pr(n-class)
            
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            
            if classID == personIdx and confidence > MIN_CONF: 
                box = detection[0:4] * np.array([W,H,W,H])
                (centerX,centerY,width,height)=box.astype('int')
                
                x = int(centerX - (width/2))
                y = int(centerY - (height/2))
                
                boxes.append([x,y,int(width),int(height)])
                centroids.append([centerX,centerY])
                confidences.append(float(confidence))
    
    idx = cv2.dnn.NMSBoxes(boxes,confidences,MIN_CONF,NMS_THRESH)
    
    if len(idx) > 0:
        
        for i in idx.flatten():
            
            (x,y) = (boxes[i][0],boxes[i][1])
            (w,h) = (boxes[i][2],boxes[i][3])

    
            r = (confidences[i],(x,y,x+w,y+h),centroids[i])
            results.append(r)
            
    return results
