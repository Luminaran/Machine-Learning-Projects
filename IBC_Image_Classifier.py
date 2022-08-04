# In this project the goal is to make an effective image classifier that can tell whether an indivudual has IBC based on 50 x 50 histology slide slices
import numpy as np
from numpy import load
from numpy import asarray
from numpy import save
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imread
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Sequential
from keras.models import Model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from os import listdir
import os
import cv2
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
import tensorflow as tf
from sklearn.metrics import classification_report
# Preparing a smaller dataset to quickly train a smaller model
DIRECTORY = r'C:\Users\Breast_Cancer_Slides\8863'
CATEGORIES = ['0','1']
IMG_SIZE = 50
data = []

for category in CATEGORIES:
    folder = os.path.join(DIRECTORY, category)
    label = CATEGORIES.index(category)
    for img in os.listdir(folder):
        img_path = os.path.join(folder, img)
        img_arr = imread(img_path)
        img_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
        data.append([img_arr, label])
# Checking for all 979 images  
len(data) 
# Makes sure photos aren't all non-cancerous followed by cancerous confusing the model       
random.shuffle(data)        
# Save data in a reusable format
X = []
y = []
for features , labels in data:
    X.append(features)
    y.append(labels)
X = np.array(X)
y = np.array(y)
save('Small_Cancer_Data_train_X', X)
save('Small_Cancer_Data_train_y', y)        
X = load('Small_Cancer_Data_train_X.npy')
y = load('Small_Cancer_Data_train_y.npy')
print(X.shape, y.shape) # Everything looks good, X is 979 images of size 50 x 50, made up of RGB colors. y is the 979 labels        
# Making the Model      
model = Sequential()# Groups a linear stack of layers into a tf.keras.Model
# Layer 1
model.add(Conv2D(256, (3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))# Pools 2 pixel by 2 pixel chunks
model.add(Dropout(0.2))
# Layer 2
model.add(Conv2D(256, (3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))
# Layer 3
model.add(Conv2D(256, (3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))# pools 2 pixel by 2 pixel chunks
model.add(Dropout(0.2))
# Output Layer
model.add(Flatten())# Converts the data into a 1-dimensional array for inputting it to the next layer, eg (2,2) becomes (4)
model.add(Dense(512, input_shape = X.shape[1:], activation = 'relu'))
model.add(Dense(2, activation = 'softmax', kernel_regularizer='l2'))# l2 regularization limits the models ability to haevily bias individual neurons
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])        
# Run the model
model.fit(X, y, epochs = 80, validation_split = 0.1) # Validation accuracy peaks around 90%, beyond that severe overfitting begins to occur
# The model is doing as well as can be expected on the limited data set, time to save it and massively expand the datset
model_path = '/Cancer_Classifier/keras_save'
model.save(model_path)
model = tf.keras.models.load_model(model_path)
# Time to prepare a new dataset, made up of a majority of the data, as well as a test data set for later use
DIRECTORY = r'C:\Users\Breast_Cancer_Slides\Train_Data'
CATEGORIES = ['0','1']
IMG_SIZE = 50
data_train = []
for category in CATEGORIES:
    folder = os.path.join(DIRECTORY, category)
    label = CATEGORIES.index(category)
    for img in os.listdir(folder):
        img_path = os.path.join(folder, img)
        img_arr = imread(img_path)
        img_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
        data_train.append([img_arr, label])
len(data_train)
random.shuffle(data_train)
# Save training data in a reusable format
X = []
y = []
for features , labels in data_train:
    X.append(features)
    y.append(labels)
X = np.array(X)
y = np.array(y)
save('Cancer_Data_train_X', X)
save('Cancer_Data_train_y', y)
X = load('Cancer_Data_train_X.npy')
y = load('Cancer_Data_train_y.npy')
print(X.shape, y.shape)# Everything looks good, X is 49515 images of size 50 x 50, made up of RGB colors. y is the 49515 labels
# Run the enlarged model
model.fit(X,y, epochs=50,validation_split = 0.2) # The model eventaully peaks at roughly 95% accuracy and 87% validation accuracy
# Save final model
model_path2 = '/Cancer_Classifier2/keras_save'
model.save(model_path2)
model = tf.keras.models.load_model(model_path2)
# Create test data
DIRECTORY = r'C:\Users\Breast_Cancer_Slides\Test_Data'
CATEGORIES = ['0','1']
IMG_SIZE = 50
data = []
for category in CATEGORIES:
    folder = os.path.join(DIRECTORY, category)
    label = CATEGORIES.index(category)
    for img in os.listdir(folder):
        img_path = os.path.join(folder, img)
        img_arr = imread(img_path)
        img_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
        data.append([img_arr, label])
len(data)
random.shuffle(data)
X = []
y = []
for features , labels in data:
    X.append(features)
    y.append(labels)
X = np.array(X)
y = np.array(y)
save('Cancer_Data_test_X', X)
save('Cancer_Data_test_y', y)
X_test = load('Cancer_Data_test_X.npy')
y_test = load('Cancer_Data_test_y.npy')
print(X.shape, y.shape)# Everything looks good, X is 25369 images of size 50 x 50, made up of RGB colors. y is the 25369 labels
# Evaluate model
test_loss, test_accuracy = model.evaluate(X_test, y_test)# 75% test accuracy shows that the model has overfitted quite a bit on the training data
