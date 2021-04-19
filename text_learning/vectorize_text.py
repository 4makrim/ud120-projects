#!/usr/bin/python

import os
import pickle
import re
import sys

sys.path.append( "../tools/" )
from parse_out_email_text import parseOutText

"""
    Starter code to process the emails from Sara and Chris to extract
    the features and get the documents ready for classification.

    The list of all the emails from Sara are in the from_sara list
    likewise for emails from Chris (from_chris)

    The actual documents are in the Enron email dataset, which
    you downloaded/unpacked in Part 0 of the first mini-project. If you have
    not obtained the Enron email corpus, run startup.py in the tools folder.

    The data is stored in lists and packed away in pickle files at the end.
"""


from_sara  = open("from_sara.txt", "rb")
from_chris = open("from_chris.txt", "rb")

from_data = []
word_data = []

### temp_counter is a way to speed up the development--there are
### thousands of emails from Sara and Chris, so running over all of them
### can take a long time
### temp_counter helps you only look at the first 200 emails in the list so you
### can iterate your modifications quicker
temp_counter = 0


for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
    for path in from_person:
        ### only look at first 200 emails when developing
        ### once everything is working, remove this line to run over full dataset
        temp_counter += 1
        #if temp_counter < 200:
        path = os.path.join('..', path[:-1].decode('utf-8'))
        #print(path)
        email = open(path, "r")
        #print(email)
        
        ### use parseOutText to extract the text from the opened email
        text = parseOutText(email)
        #print(text)
            
        rem_words = ["sara", "shackleton", "chris", "germani"]
        for word in rem_words:
            text = text.replace(word, '')

        ### append the text to word_data
        word_data.append(text)
        ### append a 0 to from_data if email is from Sara, and 1 if email is from Chris
        if name == 'sara':
            from_data.append(0)
        elif name == 'chris':
            from_data.append(1)
            
        print(temp_counter, ' :: ', text)
        email.close()

print("emails processed")
from_sara.close()
from_chris.close()

pickle.dump( word_data, open("your_word_data.pkl", "wb") )
pickle.dump( from_data, open("your_email_authors.pkl", "wb") )

### in Part 4, do TfIdf vectorization here
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(input=word_data, stop_words='english')
X = vectorizer.fit_transform(word_data)
print(len(vectorizer.get_feature_names()))

### What is word 34597 in TfIdf that you just made?
print(vectorizer.get_feature_names()[34597])

### References
### Snowball stemmer - http://www.nltk.org/howto/stem.html
### Download nltk - https://stackoverflow.com/questions/5843817/programmatically-install-nltk-corpora-models-i-e-without-the-gui-downloader
### https://www.freecodecamp.org/news/how-to-extract-keywords-from-text-with-tf-idf-and-pythons-scikit-learn-b2a0f3d7e667/


