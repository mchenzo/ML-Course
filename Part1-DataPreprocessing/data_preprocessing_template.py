# Data Preprocessing Template

# Importing the libraries
import sys
sys.path.append('/usr/local/lib/python3.6/site-packages')
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import sklearn

# Importing the dataset
dataset = pd.read_csv('/Users/michaelchen/Desktop/MachineLearningA-ZTemplateFolder/Part1-DataPreprocessing/Data.csv')
# all rows and all cols except last col
X = dataset.iloc[:, :-1].values
# all rows, only last col
Y = dataset.iloc[:, 3].values

# Taking care of missing data
# from sklearn.preprocessing import Imputer
# # create an imputer object, replace NaN values with mean of columns
# imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
# # fit imputer to columns with missing data, all rows, upper bound excluded
# imputer = imputer.fit(X[:, 1:3])
# # replace missing data in X with imputer 
# X[:, 1:3] = imputer.transform(X[:, 1:3])


# Encoding categorical data
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# labelEncoder_X = LabelEncoder()
# # encoded value of categorical variable in 1st col
# # PROBLEM: assigns values to categories; is 1 category better than another?
# X[:, 0] = labelEncoder_X.fit_transform(X[:, 0])

# specify which column is categorical
# oneHotEncoder = OneHotEncoder(categorical_features = [0])
# X = oneHotEncoder.fit_transform(X).toarray()

# # dependent variable 
# labelEncoder_Y = LabelEncoder()
# Y = labelEncoder_Y.fit_transform(Y)

# Splitting dataset into the Training set and the Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature scaling: unbias influence of 1 var on euclidean distance 
""""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
# fit scaler to training set before transforming training set (scaling)
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""
