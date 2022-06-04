# In this project the goal is to make an effective image classifier that can tell images of 120 different dog breeds apart
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
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D 
import tensorflow as tf

DIRECTORY = r'C:\Users\dog_breed\dog_breed_classifier_images'
labels = pd.read_csv('dog_breed_labels.csv')
# Looking at the labels
labels.head()
# The only needed data in labels is the dog breed
labels = labels['breed']
# This checks that my images are properly being shown
IMG_SIZE = 224
data = []
folder = os.path.join(DIRECTORY)
for img in os.listdir(folder):
        img_path = os.path.join(folder, img)
        img_arr = imread(img_path)
        img_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
        plt.imshow(img_arr)
        break
# Images are showing properly, now I'll modify all images to make them the same size
IMG_SIZE = 224
data = []
num = []
folder = os.path.join(DIRECTORY)
for img in os.listdir(folder):
        img_path = os.path.join(folder, img)
        img_arr = imread(img_path)
        img_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
        data.append([img_arr])
# Labels need to be midified from categprical catagories into one-hot numeric array's, then saved for later use
y = labels
onehot = OneHotEncoder() #Encodes categorical features as a one-hot numeric array
y = LabelEncoder().fit_transform(y) #Encodes target labels with value between 0 and n_classes-1
y = onehot.fit_transform(np.expand_dims(y, axis=1)).toarray() #Encodes categorical features as a one-hot numeric array
# Check the data to make sure everything worked
y.shape # 10222 data points in 120 catagories, everything looks good
save('Dog_Breed_Labels',y)
# Next I will save the data as numpy arrays that can be saved for later use
X = []
for i in data:
    X.append(i)
X = np.array(X)
X = np.squeeze(X) # Removes axes of length one 
X.shape # X.shape shows that X is the expected 10222 images of size 224 x 224, in RGB color
save('Dog_Breed_Images', X)
# Data and labels can now be loaded whenever without going back through all the previous code
X = load('Dog_Breed_Images.npy')
y = load('Dog_Breed_Labels.npy')
# Now to test a variety of different models for accuracy
# One layer VGG
model1 = Sequential()# groups a linear stack of layers into a tf.keras.Model
# layer 1
model1.add(Conv2D(64, (3,3), activation = 'relu',input_shape=(224 224 3)))
model1.add(MaxPooling2D((2,2)))# pools 2 pixel by 2 pixel chunks

model1.add(Flatten())# converts the data into a 1-dimensional array for inputting it to the next layer, eg (2,2) becomes (4)
model1.add(Dense(128, input_shape = X.shape[1:], activation = 'relu'))
model1.add(Dense(120, activation = 'softmax'))
model1.compile(optimizer = 'adam', loss ='categorical_crossentropy', metrics = ['accuracy'])
model1.fit(X, y, epochs = 5, validation_split = 0.2  #Accuracy numbers are currently around 1.2%, not much better than chance, a better model is needed
# Three layer VGG
model2 = Sequential()
# layer 1
model2.add(Conv2D(64, (3,3), activation = 'relu',input_shape=(224 224 3)))
model2.add(MaxPooling2D((2,2)))
#layer 2
model2.add(Conv2D(64, (3,3), activation = 'relu'))
model2.add(MaxPooling2D((2,2)))
#layer 3
model2.add(Conv2D(128, (3,3), activation = 'relu'))
model2.add(MaxPooling2D((2,2)))

model2.add(Flatten())
model2.add(Dense(128, input_shape = X.shape[1:], activation = 'relu'))
model2.add(Dense(120, activation = 'softmax'))
model2.compile(optimizer = 'adam', loss ='categorical_crossentropy', metrics = ['accuracy'])
model2.fit(X, y, epochs = 5, validation_split = 0.2) #Nt a significant iprovement
# Due to the small number of images of each dog, it will be helpful to use a pre-trained image classifier on our data
# VGG 16 layer tranfer model
model3 = VGG16(include_top=False, input_shape=(224, 224, 3))#include_top: whether to include the 3 fully-connected layers at the top of the network, note the function was treained using 224 x 224 data, so the data would need to be changed to this to make it work
# mark loaded layers as not trainable
for layer in model6.layers:
		layer.trainable = False
# add new classifier layers
flat3 = Flatten()(model3.layers[-1].output)
class3 = Dense(1024,activation='relu', kernel_initializer='he_uniform')(flat3)
output3 = Dense(120, activation='relu')(class3)
model3 = Model(inputs=model3.inputs, outputs=output3)
model3.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model3.fit(X, y, epochs = 20, validation_split = 0.2) # Training accuracy is above 10%, and validation accuracy remains low, perhaps a different model may be effective
# MobileNetV2 model with dropout
model4 = MobileNetV2(include_top=False, input_shape=(224, 224, 3))
# mark loaded layers as trainable
for layer in model4.layers:
		layer.trainable = True
# add new classifier layers
x = model4.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
x = Dense(2048, activation="relu")(x)
x = Dense(120, activation="softmax")(x)
model4= Model(inputs=model8.inputs, outputs=x)
model4.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model4.fit(X, y, epochs = 40,validation_split = 0.2) # 95.6% accuracy with 40.2% valence accuracy is a massive improvement, and by far the best model  made
# Save model for future use
model_path = '/dog_breed/model_save'
model4.save(model_path)
final_model = tf.keras.models.load_model(model_path)
