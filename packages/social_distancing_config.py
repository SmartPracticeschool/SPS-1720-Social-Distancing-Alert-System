# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 11:46:51 2020

@author: GANGASAGAR
"""

#base path to yolo directory
MODEL_PATH='yolo-coco'

MIN_CONF=0.3
NMS_THRESH=0.3

#presence of cuda
USE_GPU=False

#define minimum safe distance that people can 
# be from each other
MIN_DISTANCE = 50