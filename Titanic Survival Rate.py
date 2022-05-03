# The goal of this project is to find the model that can most accurately predict whether an individual will survive their trip on the titanic
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
# The dataset and test set
train_data = pd.read_csv('train-titanic.csv')
test_data = pd.read_csv('test-titanic.csv')
test_data_s = pd.read_csv('titanic-survival.csv')
# First look at the data to get a feeling for it
test_data.head()
test_data.info()
# A visual look at what are likely to be some of the key data columns when trying to predict survival rate
sns.countplot(x='Survived', data=train_data, hue='Pclass')# 0 = died, 1 = survived
sns.countplot(x='Survived', data=train_data, hue='Sex')
sns.distplot(train_data['Age'])
# Now the data has to be cleaned up so it can be used
train_data.isnull().sum()# This shows age is missing for a large number of people, and cabin is missing for the majority of people, with 2 missing embarked
sns.boxplot(x='Pclass', y='Age',data=train_data)
# The missing data for age needs to be replaced I'll be using the average. I could take the avg of all the ages, but since there is a clear difference between the different Pclass's, for maximum accuracy I'll set mean ages based on what Pclass the missing age data point their individual belongs to.
# I can set mean ages based on what Pclass the missing age data point their individual belongs to.
fc_mean = round(train_data[train_data['Pclass'] == 1]['Age'].mean())# 1st class age mean
sc_mean = round(train_data[train_data['Pclass'] == 2]['Age'].mean())# 2nd class age mean
tc_mean = round(train_data[train_data['Pclass'] == 3]['Age'].mean())# 3rd class age mean
def fill_na_age(cols):# This function will replace all null ages with new mean ages based on what Pclass the person's ticket was
    age = cols[0]
    pclass = cols[1]
    if pd.isnull(age):
        if pclass ==1:
            return fc_mean
        elif pclass ==2:
            return sc_mean
        elif pclass ==3:
            return tc_mean
    else:
        return age
train_data['Age'] = train_data[['Age', 'Pclass']].apply(fill_na_age, axis=1)
test_data['Age'] = test_data[['Age', 'Pclass']].apply(fill_na_age, axis=1)
t_mean = test_data['Fare'].mean()# The test data had some missing values for fare which I've averaged
test_data.fillna(value=t_mean, inplace=True)
train_data.isnull().sum()# All null ages have now been replaced
# Becuase the vast majority of cabin data is missing, trying to avg it out doesn't make much sense, we will just drop it
train_data.drop(['Cabin'],axis=1,inplace=True)# inplace = True makes the change occur automatically so I don't need to say train_data = exc.
train_data.dropna(inplace=True)# This removes the two missing values for embarked
# Now I remove columns that we don't think will be handy for finding survival rates
train_data.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)
test_data.drop(['PassengerId','Name','Ticket', 'Cabin'],axis=1,inplace=True)
train_data['Embarked'].unique()# .unique shows all the different values for the chosen data
# Now the categorical data has to be changed into numerical data using one-hot encoding
sex = pd.get_dummies(train_data['Sex'],drop_first=True)
embarked = pd.get_dummies(train_data['Embarked'],drop_first=True)
sex2 = pd.get_dummies(test_data['Sex'],drop_first=True)
embarked2 = pd.get_dummies(test_data['Embarked'],drop_first=True)
# Then I removed the old sex and embarked values and replace them with the new one-hot values
train_data.drop(['Sex','Embarked'], axis=1,inplace=True)
test_data.drop(['Sex','Embarked'], axis=1,inplace=True)
train_data = pd.concat([train_data,sex,embarked],axis=1)
test_data = pd.concat([test_data,sex2,embarked2],axis=1)
# The newly modified datasets
test_data.head()
train_data.head()
# Now the training data will be split into training and validation data sets
X = train_data.drop(['Survived'], axis=1)
y = train_data['Survived']
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size = 0.1)
#The real test data
X_Test = test_data
Y_Test = test_data_s['Survived']
#Normalizing data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_Test = scaler.transform(X_Test)
# Support-vector machines (SVMs, also support-vector networks[1]) are supervised learning models with associated learning algorithms that analyze data for classification and regression analysis
# Given a set of training examples, each marked as belonging to one of two categories, an SVM training algorithm builds a model that assigns new examples to one category or the other, making it a non-probabilistic binary linear classifier
svm = SVC()
svm.fit(X_train, y_train)
predictions = svm.predict(X_test)
# A classification report will give the model's success rate in the form of precision,recall,f1 scores, and accuarecy for 
# predicting the correct state(survival or death in this case)
print(classification_report(y_test,predictions))
# A confusion matrix is a table that is used to define the performance of a classification algorithm. A confusion matrix visualizes and summarizes the performance of a classification algorithm. 
# The confusion matrix consists of four basic characteristics (numbers) that are used to define the measurement metrics of the classifier
# 1: True Positive(TP) found in top-left
# 2: True Negative(TN) found in bottom-right
# 3: False Positive(FP) found in bottom-left, aka type I error
# 4 False Negative(FN) found in top-right, aka type II error
print(confusion_matrix(y_test,predictions))
# The prediction rate may be improved by finding better C and gamma values for the model
param_grid = {'C':[0.5,1,10,50,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001,0.00001,0.000001]}
# The C parameter trades off correct classification of training examples against maximization of the decision function’s margin. For larger values of C, a smaller margin will be accepted if the decision function is better at classifying all training points correctly.
#A lower C will encourage a larger margin
#The gamma parameter defines how far the influence of a single training example reaches, with low values meaning ‘far’ and high values meaning ‘close’
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=0)# Exhaustively considers all parameter combinations
grid.fit(X_train,y_train)
grid_predictions = grid.predict(X_test)
# Results are slightly improved
print(classification_report(y_test,grid_predictions))
print(confusion_matrix(y_test,grid_predictions))
# There are many possible models to try that may give better results than the SVC model, several were attempted below
#Linear Regression
lr = LogisticRegression()
lr.fit(X_train,y_train)
lr_predictions = lr.predict(X_test)
print(classification_report(y_test,lr_predictions))# slightly lower results than SVC
print(confusion_matrix(y_test,lr_predictions))
#K nearest Neighbor
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
knn_predictions = knn.predict(X_test)
print(classification_report(y_test,knn_predictions))# slightly lower results than SVC and logisitc regression, but we can do better by choosing n-neigbors
print(confusion_matrix(y_test,knn_predictions))
# The following code will find the best number of n-neighbors to maximize the models effectiveness
error_list = []
for i in range(1,40):
    knn2 = KNeighborsClassifier(n_neighbors = i)
    knn2.fit(X_train,y_train)
    knn2_predictions = knn2.predict(X_test)
    error_list.append(np.mean(knn2_predictions != y_test))
    plt.plot(range(1,40),error_list)
np.argmin(error_list)# Best result is n-neighbors of 2
knn = KNeighborsClassifier(n_neighbors = 2)
knn.fit(X_train,y_train)
knn_predictions = knn.predict(X_test)
print(classification_report(y_test,knn_predictions))# slightly lower results than SVC, but now better than logisitc regression
print(confusion_matrix(y_test,knn_predictions))
#Decision Tree Classifier
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
dt_predictions = dt.predict(X_test)
print(classification_report(y_test,dt_predictions))# Worst model so far
print(confusion_matrix(y_test,dt_predictions))
# Random Forest Classifier
# First I will find a good n_estimator
error_list2 = []
for i in range(1,100):
    rfc = RandomForestClassifier(n_estimators = i)
    rfc.fit(X_train,y_train)
    rfc_predictions = rfc.predict(X_test)
    error_list2.append(np.mean(rfc_predictions != y_test))
 plt.plot(range(1,100),error_list2)# highly variable results
np.argmin(error_list)# In this case the result was 31
rfc = RandomForestClassifier(n_estimators=31)
rfc.fit(X_train,y_train)
rfc_predictions = rfc.predict(X_test)
print(classification_report(y_test,rfc_predictions))# still worse than SVC in this case
print(confusion_matrix(y_test,rfc_predictions))
# In the end SVC was found to be the most accurate model for the validation data, so it will be used on the true test data
grid_predictions2 = grid.predict(X_Test)
print(classification_report(Y_Test,grid_predictions2))
print(confusion_matrix(Y_Test,grid_predictions2))
# Weighted averages for my model are 0.88 precision, 0.88 recall, and 0.87 accuracy

