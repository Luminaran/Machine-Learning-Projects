# The K means-clustering system is as follows;
#1:Place k random centroids for the initial clusters
#2:Assign data samples to the nearest centroid
#3:Update centroids based on the above-assigned data samples
#4: repeat steps 2-3 until convergence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
print(iris.data[0:10])# each row is 4 datapoints about a specific iris
print(iris.target)# as this is practice data you can see the real answer for which of the 3 iris types each datapoint came from
print(iris.DESCR)# A description of the data

# A visulization of some of the data
samples = iris.data
x = samples[:,0]
y = samples[:,1]
sepal_length_width = np.array(list(zip(x, y)))
plt.scatter(x, y, alpha=0.5)
 
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
 
plt.show()
#K means step 1:Place k random centroids for the initial clusters.
# as we know there are 3 flower species, we can make k = 3
k = 3
centroids_x = np.random.uniform(min(x), max(x), size=k)# the random x value for starting centroids
centroids_y = np.random.uniform(min(y), max(y), size=k)# the random y value for starting centroids
centroids = np.array(list(zip(centroids_x,centroids_y)))# creates an array with your centroids
plt.scatter(x,y, alpha=0.5)
plt.scatter(centroids_x, centroids_y)
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')# orange dots are the random starting centroids

#K means step 2: Assign data samples to the nearest centroid
def distance(a,b):
  one = (a[0] - b[0]) ** 2
  two = (a[1] - b[1]) ** 2
  distance = (one+two) ** 0.5
  return distance

for i in range(len(samples)):
    distances[0] = distance(sepal_length_width[i], centroids[0])
    distances[1] = distance(sepal_length_width[i], centroids[1])
    distances[2] = distance(sepal_length_width[i], centroids[2])
    cluster = np.argmin(distances)
    labels[i] = cluster
    print(labels)# each data point has now been asigned to one of the 3 clusters based on whichever distances was the lowest

#K means step 3: Update centroids based on the above-assigned data samples
from copy import deepcopy
centroids_old = deepcopy(centroids)
#A deep copy constructs a new compound object and then, recursively, inserts copies into it of the objects found in the original
for i in range(k):
  points = []
  for j in range(len(sepal_length_width)):
    if labels[j] == i:
      points.append(sepal_length_width[j])
    centroids[i] = np.mean(points, axis=0)
 
#K means step 4 repeat 2-3 until convergence
error = np.zeros(3)

error[0] = distance(centroids[0], centroids_old[0])
error[1] = distance(centroids[1], centroids_old[1])

while error.all() != 0:

  # Step 2: Assign samples to nearest centroid

  for i in range(len(samples)):
    distances[0] = distance(sepal_length_width[i], centroids[0])
    distances[1] = distance(sepal_length_width[i], centroids[1])
    distances[2] = distance(sepal_length_width[i], centroids[2])
    cluster = np.argmin(distances)
    labels[i] = cluster

    #Step 3: Update centroids

  centroids_old = deepcopy(centroids)

  for i in range(3):
    points = [sepal_length_width[j] for j in range(len(sepal_length_width)) if labels[j] == i]
    centroids[i] = np.mean(points, axis=0)

  error[0] = distance(centroids[0], centroids_old[0])
  error[1] = distance(centroids[1],   centroids_old[1])
  error[2] = distance(centroids[2], centroids_old[2])

colors = ['r', 'g', 'b']

for i in range(k):
  points = np.array([sepal_length_width[j] for j in range(len(samples)) if labels[j] == i])
  plt.scatter(points[:, 0], points[:, 1], c=colors[i], alpha=0.5)

plt.scatter(centroids[:, 0], centroids[:, 1], marker='D', s=150)

plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')

plt.show()

error[2] = distance(centroids[2], centroids_old[2])
  
