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

# Cleaning the texts


# Creating the Bag of Words model


# Splitting the dataset into the Training set and Test set


# Fitting Naive Bayes to the Training set


# Predicting the Test set results


# Making the Confusion Matrix

