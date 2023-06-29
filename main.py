import cv2
import numpy as np

image=cv2.imread("technology-792180_1280.jpg")

class_name=[]


class_file='coco.names'

with open(class_file,'rt')as f:
    
    class_name=f.read().rstrip('\n').split('\n')
    #copier  class_name dans un fichier :


conf_path='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'

weigt_path='frozen_inference_graph.pb'


net=cv2.dnn_DetectionModel(conf_path,weigt_path)

net.setInputSize(320, 230)
net.setInputScale(1.0/127.5)
#net.setInputMean(12)
net.setInputSwapRB(True)

class_ids,conf,bbox=net.detect(image,confThreshold=0.6)
if len(class_ids) > 0:
    for ids,confidence,box in zip(class_ids.flatten(),conf.flatten(),bbox):
        
        cv2.rectangle(image,box,color=(0,255,0),thickness=2)
        cv2.putText(image,class_name[ids-1],(box[0]+10,box[1]+20),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),thickness=2)

cv2.imshow('images',image)


cv2.waitKey(0)