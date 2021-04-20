#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl" 
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "rb"))
authors = pickle.load( open(authors_file, "rb") )



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(word_data, authors, test_size=0.1, random_state=42)




from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()

### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]

### Fit the decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
clf.fit(features_train, labels_train)
print (clf.score(features_test, labels_test))
### The accuracy of the decision tree is 0.95
### The test performance has an eccuracy much higher than it is expected to be. 
### If we are overfitting, then the test performance should be relatively low.


### Identify the most powerful feature 
### What's the importance of the most important feature? 
### What is the number of this feature? Use the index returned in output 
### to print the array element at that index - lat[output[0]]
 
lat = [i for i in clf.feature_importances_]
def condition(x): return x > 0.2
output = [idx for idx, element in enumerate(lat) if condition(element)]
print("output:", output)
for i in output:
    print("importance:",lat[i])

### There seems to be only one value here from sara chris emails. 
### It is 33614 with an imporance of 0.765
### We have determined the element at index 33614 is an outlier.

### What is the word that is causing the most discrimination? 
for i in output:
    print("featue:",vectorizer.get_feature_names()[i])

from sklearn.feature_selection import SelectPercentile, f_classif
selector = SelectPercentile(f_classif, percentile=10)
df = selector.fit(features_train, labels_train)

# features_train_transformed = selector.transform(features_train).toarray()
# features_test_transformed = selector.transform(features_test).toarray()
features_train_transformed = selector.transform(features_train)
features_test_transformed = selector.transform(features_test)


print(features_train_transformed.shape)
print("no. of Chris training emails:", sum(labels_train))
print("no. of Sara training emailsL", len(labels_train)-sum(labels_train))

