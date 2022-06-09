# In this project the goal is to create a model that can accurately predict the tree species of a forest cover
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.layers import Dropout
from tensorflow.keras.layers import InputLayer
import tensorflow as tf

data = pd.read_csv('cover_data.csv')
# Looking at the data
data.head()
data.info()
data.isnull().sum()# There is no missing data
# Split data into X and y variables
y = data['class']
data.drop(['class'],axis=1,inplace=True)
X = data
# Prepare data for model
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size = 0.2)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# First Model
model1 = Sequential()
model1.add(InputLayer(input_shape = (X_train.shape[1])))
# Layer 1
model1.add(Dense(128, activation = 'relu'))
model1.add(Dense(256, activation = 'relu'))
model1.add(Dropout(0.1))
# Layer 2
model1.add(Dense(128, activation = 'relu'))
model1.add(Dense(256, activation = 'relu'))
model1.add(Dropout(0.1))
# Layer 3
model1.add(Dense(256, activation = 'relu'))
model1.add(Dense(512, activation = 'relu'))
model1.add(Dropout(0.1))
# Output Layer
model1.add(Dense(8, activation = "softmax"))
model1.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
history = model1.fit(X_train, y_train, epochs = 50, validation_split = 0.1) # accuracy of 50 epochs was ~ 91% for both training and validation data

# Plot of Accuracy
fig = plt.figure(figsize = (15,10))
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['accuracy'])
ax1.plot(history.history['val_accuracy'])
ax1.set_title('model accuracy')
ax1.set_ylabel('accuracy')
ax1.set_xlabel('epoch')
ax1.legend(['train', 'validation'], loc='upper left')# The model seems good, now to increase the number of layers in the model to see if accuracy can be improved

# Second Model
model2 = Sequential()
model2.add(InputLayer(input_shape = (X_train.shape[1])))
# Layer 1
model2.add(Dense(512, activation = 'relu'))
model2.add(Dense(1024, activation = 'relu'))
model2.add(Dropout(0.2))
# Layer 2
model2.add(Dense(512, activation = 'relu'))
model2.add(Dense(1024, activation = 'relu'))
model2.add(Dropout(0.2))
# Layer 3
model2.add(Dense(1024, activation = 'relu'))
model2.add(Dense(2048, activation = 'relu'))
model2.add(Dropout(0.25))
# Output Layer
model2.add(Dense(8, activation = "softmax"))
model2.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
history2 = model2.fit(X_train, y_train, epochs = 4, validation_split = 0.1, batch_size=512)# ~94% accuracy after ~50 epochs

# Model 2 was the better model, and so will be saved for later use
model_path = '/forest_classifier/keras_save'
model2.save(model_path)
