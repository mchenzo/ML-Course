# Natural Language Processing

# Importing the libraries
import sys
sys.path.append('/usr/local/lib/python3.6/site-packages')
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import sklearn

# Importing the dataset, use the tsv file because the delimiter is new line not comma
dataset = pd.read_csv('/Users/michaelchen/Desktop/ML-Course/Part7-NLP/Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts; removing numbers, punctuation, reducing different tenses to root word (loved --> love)
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][0])                             # replace all non-alpha w/ spaces
review = review.lower()                                                             # reduce to lower case
review = review.split()                                                             # split into words
ps = PorterStemmer()
review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]   # include the word in review if it is not in the stopwords list
review = ' '.join(review)

# Creating the Bag of Words model


# Splitting the dataset into the Training set and Test set


# Fitting Naive Bayes to the Training set


# Predicting the Test set results


# Making the Confusion Matrix

