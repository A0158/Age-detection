import numpy as np
import os
import matplotlib.pyplot as plt
from imutils import paths
import cv2 as cv

from tensorflow.keras.applications import MobileNetV2 
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model

from imutils.video import VideoStream
import imutils
model = load_model('model3.model')
def detect_age(frame, faceNet, model):
    #Take dimensions to make blob
    (h,w) = frame.shape[:2]
    blob = cv.dnn.blobFromImage(frame, 1.0, (300,300), (104.0, 177.0, 123.0))
    
    faceNet.setInput(blob)
    detections = faceNet.forward()
    
    #initialise the list of faces, their correspondng locations and list of predictiob
    faces = []
    locs = []
    preds = []
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0,0,i,2]
        
        if confidence > 0.5:
            #coordinates of x and y
            box = detections[0,0,i,3:7]*np.array([w,h,w,h])
            (startX, startY, endX, endY) = box.astype('int')
            
            #ensure the bounding boxes fall within the dimension of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w-1, endX), min(h-1, endY))
            
            #convert ROI from BGR to RGB channel, resize ut to 224, 224 and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv.cvtColor(face, cv.COLOR_BGR2RGB)
            face = cv.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            
            faces.append(face)
            locs.append((startX, startY, endX, endY))
            
            
        if len(faces)>0:
            faces = np.array(faces, dtype = 'float32')
            preds = model.predict(faces, batch_size = 12)
            
        return (locs, preds)

vs = VideoStream(src = 0).start()

while True:
    #resize the frames
    frame = vs.read()
    frane = imutils.resize(frame, width = 400)
    
    #detect faces and age
    (locs, preds) = detect_age(frame, faceNet, model)
    
    #loop over detected face 
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (major, minor) = pred
        
         #color and lobel of the bonding box and text
        label = 'ADULT' if major>minor else 'MINOR'
        color = (0,255, 0) if label == 'ADULT' else (0, 0, 255)
            
        #display the label and bonding boxes
        cv.putText(frame, label, (startX, startY-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        cv.rectangle(frame,(startX, startY), (endX, endY), 2)
            
    cv.imshow('Frame', frame)
    key = cv.waitKey(1) & 0xFF
    
    if key ==ord('q'):
        break
    
cv.destroyAllWindows
vs.stop()