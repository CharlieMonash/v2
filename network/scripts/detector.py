import os 
import time
from xml.etree.ElementTree import XML

#import cmd_printer
import numpy as np
import torch
#from args import args
#from res18_skip import Resnet18Skip
import cv2
import json

class Detector:
    def __init__(self, ckpt, use_gpu=False):
        #Loading the custom Yolov5 model
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=ckpt, force_reload=True)

        if torch.cuda.torch.cuda.device_count() > 0 and use_gpu:
            self.use_gpu = True
            self.model = self.model.cuda()
        else:
            self.use_gpu = False
        #The model is then set to evaluation mode
        self.model = self.model.eval()

    def detect_single_image(self, np_img):
        #With Yolo no need for any conversions model will handle it
        tick = time.time()
        with torch.no_grad():
            pred = self.model.forward(np_img)
        dt = time.time() - tick
        print(f'Inference Time {dt:.2f}s, approx {1/dt:.2f}fps', end="\r")
        #Getting the coordinates for the bounding box produced by Yolo
        num_preds = len(pred.pandas().xyxy[0])
        #Making a result array to return the predictions
        pred_results = np.zeros((num_preds,5))
        #Going throuch prediction and gettring the bounding box and class prediction
        #file_result = open('fruit_estimates/fruit_boxes.txt', 'a')
        results = []
        for i in range(num_preds):
            #Get the class
            predic_class = int(pred.pandas().xyxy[0]["class"][i])
            probability = pred.pandas().xyxy[0]['confidence'][i]
            #Get the corners of the bounding box
            xl = pred.pandas().xyxy[0]['xmin'][i]
            xu = pred.pandas().xyxy[0]['xmax'][i]
            yl = pred.pandas().xyxy[0]['ymin'][i]
            yu = pred.pandas().xyxy[0]['ymax'][i]
            pred_results[i,:] = np.array([predic_class,xl,xu,yl,yu])
            #Writing the bounding boxes to a text file 
            results.append({'xmin':xl, 'ymin':yl, 'xmax':xu, 'ymax':yu, 'name':pred.pandas().xyxy[0]['name'][i], 'class': predic_class})
        #Save all the predictions to a file with the boxes
        boxes = pred.pandas().xyxy[0]
        return np.squeeze(pred.render()),np.squeeze(pred.render()),results,boxes


