import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

df = pd.read_csv('life_expectancy.csv')
#print(df.head())
#print(df.describe())
df = df.drop(['Country'], axis = 1)
#print(df.head())
labels = df.iloc[:, -1]#select all the rows (:), and access the last column (-1)
print(labels)
features = df.iloc[:, 0:-1]
#turns catagories into numerical, columns using one hot encoding
features = pd.get_dummies(features)

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.20, random_state=23)

numerical_features = features.select_dtypes(include=['float64', 'int64'])
numerical_columns = numerical_features.columns
 
ct = ColumnTransformer([("only numeric", StandardScaler(), numerical_columns)], remainder='passthrough')

features_train_scaled = ct.fit_transform(features_train)
features_test_scaled = ct.transform(features_test)

model = Sequential()
my_input = InputLayer(input_shape=(features.shape[1]))
model.add(my_input)
model.add(Dense(32, activation = "relu"))
model.add(Dense(1))
#print(model.summary())

opt = Adam(learning_rate = 0.02)
model.compile(loss = 'mse', metrics = ['mae'], optimizer = opt)

model.fit(features_train_scaled, labels_train, epochs = 30, batch_size = 1, verbose = 1)

res_mse, res_mae = model.evaluate(features_test_scaled, labels_test, verbose = 0)

#print(res_mse, res_mae)


