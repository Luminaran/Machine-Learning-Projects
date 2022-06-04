# In this project the goal is to make an effective image classifier that can tell images of dogs apart from images of cats
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
# Preparing the Data
DIRECTORY = r'C:\Users\train_dog_cat'
CATEGORIES = ['cat', 'dog']
IMG_SIZE = 150
data = []
# Labeling all data as either a cat or dog image, and making all the images into 150 x 150 pixel images
for category in CATEGORIES:
    folder = os.path.join(DIRECTORY, category)
    label = CATEGORIES.index(category)
    for img in os.listdir(folder):
        img_path = os.path.join(folder, img)
        img_arr = imread(img_path)
        img_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
        data.append([img_arr, label])
len(data)# Makes sure we have all 25,000 images
random.shuffle(data)# This makes it so all cat photos don't come first, which would confuse the machine learning programs we use later
data[0]# the zero at the end shows that this image is a cat
# Splitting the data and labels
X = []
y = []
for features , labels in data:
    X.append(features)
    y.append(labels)
X = np.array(X)
y = np.array(y)
# With the data saved I don't need to rerun all my previous work evry time I want to work on this project
save('Dog_Cat_train_X', X)
save('Dog_Cat_train_y', y)
# Now the data can be loaded whenever it is needed
X = load('Dog_Cat_train_X.npy')
y = load('Dog_Cat_train_y.npy')
print(X.shape, y.shape)# Checks to see if the data looks how it should, X should be 25,000 images of size 150 x 150, made up of RGB colors. y should be 25,000 labels
# Building the model

# One-layer VGG
model1 = Sequential()# groups a linear stack of layers into a tf.keras.Model
# layer 1
model1.add(Conv2D(64, (3,3), activation = 'relu',input_shape=(150, 150, 3)))
model1.add(MaxPooling2D((2,2)))# pools 2 pixel by 2 pixel chunks

model1.add(Flatten())# converts the data into a 1-dimensional array for inputting it to the next layer, eg (2,2) becomes (4)
model1.add(Dense(128, input_shape = X.shape[1:], activation = 'relu'))
model1.add(Dense(2, activation = 'softmax'))
model1.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model1.fit(X, y, epochs = 5, validation_split = 0.1)# This model gives us an accuracy of 0.92 on the training, but only 0.66 on the validation set, showing some serious overfitting

# Two-layer VGG
model2 = Sequential()
model2.add(Conv2D(64, (3,3), activation = 'relu'))
model2.add(MaxPooling2D((2,2)))
#layer 2
model2.add(Conv2D(64, (3,3), activation = 'relu'))
model2.add(MaxPooling2D((2,2)))

model2.add(Flatten())
model2.add(Dense(128, input_shape = X.shape[1:], activation = 'relu'))
model2.add(Dense(2, activation = 'softmax'))
model2.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model2.fit(X, y, epochs = 5, validation_split = 0.1, batch_size=32)# This gives us an accuracy of 0.98 on the training, but only 0.77 on the validation set, better but still some serious overfitting

# Three-layer VGG
model3 = Sequential()
# layer 1
model3.add(Conv2D(64, (3,3), activation = 'relu',input_shape=(150, 150, 3)))
model3.add(MaxPooling2D((2,2)))
#layer 2
model3.add(Conv2D(64, (3,3), activation = 'relu'))
model3.add(MaxPooling2D((2,2)))
#layer 3
model3.add(Conv2D(128, (3,3), activation = 'relu'))
model3.add(MaxPooling2D((2,2)))

model3.add(Flatten())
model3.add(Dense(128, input_shape = X.shape[1:], activation = 'relu'))
model3.add(Dense(2, activation = 'softmax'))
model3.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])# This gives us an accuracy of 0.84 on the training and 0.77 on the validation set, better but more improvements can be made

#Dropout Regularization is a way to increase the robustness of a model. Essentially droput randomly removes some number of layer outputs. 
#This has the effect of making the layer look-like and be treated-like a layer with a different number of nodes and connectivity to the prior layer. 
#In effect, each update to a layer during training is performed with a different “view” of the configured layer.
#Dropout has the effect of making the training process noisy, forcing nodes within a layer to probabilistically take on more or less responsibility for the inputs.
#Dropout breaks-up situations where network layers co-adapt to correct mistakes from prior layers, in turn making the model more robust.

# 3 layer VGG with dropout
model4 = Sequential()
# layer 1
model4.add(Conv2D(64, (3,3), activation = 'relu',input_shape=(150, 150, 3)))
model4.add(MaxPooling2D((2,2))
model4.add(Dropout(0.2))
#layer 2
model4.add(Conv2D(64, (3,3), activation = 'relu'))
model4.add(MaxPooling2D((2,2)))
model4.add(Dropout(0.2))
#layer 3
model4.add(Conv2D(128, (3,3), activation = 'relu'))
model4.add(MaxPooling2D((2,2)))
model4.add(Dropout(0.2))
           
model4.add(Flatten())
model4.add(Dense(128, input_shape = X.shape[1:], activation = 'relu'))
model4.add(Dropout(0.4))
model4.add(Dense(2, activation = 'softmax'))
model4.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model4.fit(X, y, epochs = 8, validation_split = 0.1) # As we expect less overfitting with dropout, a few extra epochs was added. This led to 0.83 accuracy on the training data and 0.82 on the validation data

# Image Data Augmentation: As a general rule more images for a model to train on = better models. Image Augmentation  modifies existing images to create 'new' images to test
train_datagen = ImageDataGenerator(rotation_range=10,  # rotation
                                   width_shift_range=0.1,  # horizontal shift
                                   zoom_range=0.1,)  # zoom
# 3 layer VGG with dropout and Image Augmentation
model5 = Sequential()
# layer 1
model5.add(Conv2D(64, (3,3), activation = 'relu',input_shape=(150, 150, 3)))
model5.add(MaxPooling2D((2,2))
model5.add(Dropout(0.2))
#layer 2
model5.add(Conv2D(64, (3,3), activation = 'relu'))
model5.add(MaxPooling2D((2,2)))
model5.add(Dropout(0.2))
#layer 3
model5.add(Conv2D(128, (3,3), activation = 'relu'))
model5.add(MaxPooling2D((2,2)))
model5.add(Dropout(0.2))
           
model5.add(Flatten())
model5.add(Dense(128, input_shape = X.shape[1:], activation = 'relu'))
model5.add(Dropout(0.4))
model5.add(Dense(2, activation = 'softmax'))
model5.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
train_x, val_x, train_y, val_y = train_test_split(X, y, test_size = 0.1)                              
model5.fit(train_datagen.flow(train_x, train_y), epochs = 50, validation_data =(val_x,val_y))# This dramatically increasaes the number of epochs needed, however the model eventually reaches an accuracy of 0.85 on both the training and validation data
# In the end my VGG three layer with dropout and Image Augmentation is the best, so thats the one I'll save
model_path = '/dog_cat/model_save'
model5.save(model_path)# 0.85 accuracy
