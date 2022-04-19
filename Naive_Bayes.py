import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer



emails = fetch_20newsgroups(categories=['rec.sport.baseball', 'rec.sport.hockey'])

train_emails = fetch_20newsgroups(categories = ['rec.sport.baseball', 'rec.sport.hockey'], subset = 'train', shuffle = True, random_state = 108)
test_emails = fetch_20newsgroups(categories = ['rec.sport.baseball', 'rec.sport.hockey'], subset = 'test', shuffle = True, random_state = 108)
#This breaks our data into two groups, one to train with and one to see how good our algorithim is

counter = CountVectorizer()
counter.fit(test_emails.data + train_emails.data)
train_counts = counter.transform(train_emails.data)
test_counts = counter.transform(test_emails.data)
#The fit method is calculating the mean and variance of each of the features present in our data. 
#The transform method is transforming all the features using the respective mean and variance

classifier = MultinomialNB()
classifier.fit(train_counts, train_emails.target)
print(classifier.score(test_counts, test_emails.target))
# Now our created naive bayes classifier will show how well it can tell apart hocky and soccer emails

# Lets look at two far more different claculators using the same program to see how accurate we can be
emails = fetch_20newsgroups(categories=['talk.politics.guns', 'comp.windows.x'])
train_emails = fetch_20newsgroups(categories = ['talk.politics.guns', 'comp.windows.x'], subset = 'train', shuffle = True, random_state = 108)
test_emails = fetch_20newsgroups(categories = ['talk.politics.guns', 'comp.windows.x'], subset = 'test', shuffle = True, random_state = 108)

counter = CountVectorizer()
counter.fit(test_emails.data + train_emails.data)
train_counts = counter.transform(train_emails.data)
test_counts = counter.transform(test_emails.data)

classifier = MultinomialNB()
classifier.fit(train_counts, train_emails.target)
print(classifier.score(test_counts, test_emails.target))
# Unsurpisingly emails about gun politics are much easier to tell apart from windows programming emails than soccer and baseball emails are from each other
