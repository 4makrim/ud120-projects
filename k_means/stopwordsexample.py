# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 01:40:08 2021

@author: maya_
"""

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
string1 = "Hi Katie the self driving car will be late Best Sebastian"
string2 = "Hi Sebastian the machine leaning class will be great great great Best Katie"
string3 = "Hi Katie the machine learning class will be most excellent"
email_list = [string1, string2, string3]
bag_of_words = vectorizer.fit(email_list)
print(vectorizer.vocabulary_.get('great'))


### Getting the list of english stop words that we can remove 
### from the email text to focus on words of interest.
from nltk.corpus import stopwords
sw = stopwords.words('english')
print(sw[100])

### The nltk.download() command executed on the console interpreter
### downloaded the nltk all-corpora to C:\Users\username\AppData\Roaming\nltk_data