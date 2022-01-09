# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 16:59:28 2022

@author: anirudh
"""
import numpy as np

import time
import cv2
import os
import io
from PIL import Image
import glob


confthres = 0.1
nmsthres = 0.1
yolo_path = './'
path = r'results'






def get_labels(labels_path):
    lpath=os.path.sep.join([yolo_path, labels_path])
    LABELS = open(lpath).read().strip().split("\n")
    return LABELS


def get_weights(weights_path):
    weightsPath = os.path.sep.join([yolo_path, weights_path])
    return weightsPath

def get_config(config_path):
    configPath = os.path.sep.join([yolo_path, config_path])
    return configPath

def load_model(configpath,weightspath):
    print("loading yolo")
    net = cv2.dnn.readNetFromDarknet(configpath, weightspath)
    return net





def get_predection(image,net,LABELS):
    (H, W) = image.shape[:2]

    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    print(layerOutputs)
    end = time.time()

    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    
    boxes = []
    confidences = []
    classIDs = []


    for output in layerOutputs:
 
        for detection in output:
            
            scores = detection[5:]
            
            classID = np.argmax(scores)
          
            confidence = scores[classID]

            
            if confidence > confthres:
                #
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confthres,
                            nmsthres)
    l = 1

    
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            c_i = image[y:y+h , x:x+w]
            if LABELS[classIDs[i]] != "vehicle":               
                cv2.imwrite(os.path.join(path,LABELS[classIDs[i]]+str(l)+'.jpg'),c_i)
                l=l+1 # to take care of more than one license plate
            print(LABELS[classIDs[i]])



        
                 
labelsPath="yolov5/data.names"
cfgpath="yolov5/yolov3_custom.cfg"
wpath="yolov5/yolov3_custom_1000.weights"
Lables=get_labels(labelsPath)
CFG=get_config(cfgpath)
Weights=get_weights(wpath)
nets=load_model(CFG,Weights)


def main():
    os.chmod(path, 777)
    files = glob.glob(path)
    for f in files:       
        os.remove(f)
    image = cv2.imread("testimages/ccc1a2d44a290368_jpg.rf.8e8d6f87f1d1327caf5f80e0c96cdad9.jpg")
    npimg=np.array(image)
    image=npimg.copy()
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    get_predection(image,nets,Lables)


    

        
if __name__ == '__main__':
    main()
