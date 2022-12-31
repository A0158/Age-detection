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
dataset = '/Users/kunal/Desktop/Age-Detection/dataset'
imagePaths = list(paths.list_images(dataset))

data = []
labels = []

for i in imagePaths:
    label = i.split(os.path.sep)[-2]
    labels.append(label)
    image = load_img(i, target_size = (224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    data.append(image)

data = np.array(data, dtype = 'float32')
labels = np.array(labels)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
train_X, test_X, train_Y, test_Y = train_test_split(data, labels, test_size = 0.20, random_state = 10, stratify = labels)
baseModel = MobileNetV2(weights='imagenet', include_top=False, input_tensor= Input(shape = (224,224,3)))
headModel = baseModel.output
headModel = AveragePooling2D(pool_size = (5,5))(headModel)
headModel = Flatten(name = 'Flatten')(headModel)
headModel = Dense(128, activation = 'relu')(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation = 'softmax')(headModel)

model = Model(inputs = baseModel.input, outputs = headModel)

for layer in baseModel.layers:
    layer.trainable = False
model.summary()
learning_rate = 0.01
Epochs = 10
batch_size = 12

opt = Adam(lr=learning_rate, decay = learning_rate/Epochs)

model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'])

H = model.fit(
    train_X, train_Y, batch_size = batch_size,
    steps_per_epoch=len(train_X)//batch_size,
    validation_data=(test_X, test_Y),
    validation_steps=len(test_X)//batch_size,
    epochs = Epochs
)
model.save('model3.model')