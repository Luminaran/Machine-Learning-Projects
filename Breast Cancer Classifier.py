from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
breast_cancer_data = load_breast_cancer()

breast_cancer_data.feature_names
breast_cancer_data.target
breast_cancer_data.target_names# our goal is to classify tumors as either malignant or benign

training_data, validation_data, training_labels, validation_labels = train_test_split(breast_cancer_data.data,
                 breast_cancer_data.target,
                 test_size=0.2,
                 random_state=100)

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(training_data, training_labels)
classifier.score(validation_data, validation_labels)

# Using a for loop to find the ideal k value
for k in range(1, 300):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(training_data, training_labels)
    classifier.score(validation_data, validation_labels)
    print(classifier.score(validation_data, validation_labels))# score peaks ~25
    
    accuracies = []
for k in range(1, 301):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(training_data, training_labels)
    accuracies.append(classifier.score(validation_data, validation_labels))
    
k_list = range(1, 301)
plt.plot(k_list, accuracies)
plt.xlabel('K')
plt.ylabel('Validation Accuracy')
plt.title('Breast Cancer Classifier Accuracy')
plt.show()
