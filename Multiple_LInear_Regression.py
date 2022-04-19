import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

streeteasy = pd.read_csv("https://raw.githubusercontent.com/sonnynomnom/Codecademy-Machine-Learning-Fundamentals/master/StreetEasy/manhattan.csv")

df = pd.DataFrame(streeteasy)

x = df[['bedrooms', 'bathrooms', 'size_sqft', 'min_to_subway', 'floor', 'building_age_yrs', 'no_fee', 'has_roofdeck','has_washer_dryer', 'has_doorman', 'has_elevator', 'has_dishwasher', 'has_patio', 'has_gym']]

y = df[['rent']]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state=6)
mlr = LinearRegression()
mlr.fit(x_train.values, y_train.values)
y_predict = mlr.predict(x_test)

sonny_apartment = [[1, 1, 620, 16, 1, 98, 1, 0, 1, 0, 0, 1, 1, 0]]# each x value, so it has 1 bed. 1 bath, 620sq ft exc.
predict = mlr.predict(sonny_apartment)
print(predict)

# Data vs Predicted
lm = LinearRegression()

model=lm.fit(x_train, y_train)

y_predict = lm.predict(x_test)
plt.scatter(y_test, y_predict)
plt.xlabel("True Rent Price")
plt.ylabel("Expected Rent Price")
plt.show()

#.fit gives the coefficents, which are handy for findng which variables are most important, either highly pos or neg
df = pd.DataFrame(streeteasy)

x = df[['bedrooms', 'bathrooms', 'size_sqft', 'min_to_subway', 'floor', 'building_age_yrs', 'no_fee', 
        'has_roofdeck', 'has_washer_dryer', 'has_doorman', 'has_elevator', 'has_dishwasher', 'has_patio', 'has_gym']]

y = df[['rent']]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state=6)

mlr = LinearRegression()

model=mlr.fit(x_train, y_train)

y_predict = mlr.predict(x_test)
print(mlr.coef_)# these make a lot of sense, each individual bathroom has a major effect(~1200) while each individual sq ft only has about ~4

# eyeballing some correlations
plt.scatter(df[['min_to_subway']], df[['rent']], alpha=0.4)# Not a clear correlation
plt.scatter(df[['size_sqft']], df[['rent']], alpha=0.4)# A clear correlation

# A basic way to evaluate model accuaracy is with residual analysis using R^2, calculated as 1 - (U/V)
# U is the residual sum of squares = ((y - y_predict) ** 2).sum()
# V is the total sum of squares = ((y - y.mean()) ** 2).sum()
print(mlr.score(x_train, y_train))
print(mlr.score(x_test, y_test))

# The model can be improved by removing features without strong correlations
x = df[['bedrooms', 'bathrooms', 'size_sqft', 'no_fee', 
        'has_roofdeck', 'has_washer_dryer', 'has_doorman', 'has_elevator', 'has_dishwasher',]]

y = df[['rent']]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state=6)

lm = LinearRegression()

model = lm.fit(x_train, y_train)
y_predict= lm.predict(x_test)
print("Train score:")
print(lm.score(x_train, y_train))

print("Test score:")
print(lm.score(x_test, y_test))# This version has 5 fewer features but almost identical final values
