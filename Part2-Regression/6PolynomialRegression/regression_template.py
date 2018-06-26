# Polynomial Regression

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
dataset = pd.read_csv('/Users/michaelchen/Desktop/MachineLearningA-ZTemplateFolder/Part2-Regression/6PolynomialRegression/Position_Salaries.csv')
# ensure that X is a matrix, still only including second column
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
# With only 10 data points, splitting the data set is counter productive
# Use entire set to train model
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
# same linear regression library, handles feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Fitting the Model to the dataset


# Visualizing Regression Results
plt.scatter(X, Y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Title')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.show()

# Visualizing Regression Results in High Resolution
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Title')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.show()

# Nonlinear Model Prediction
y_pred = regressor.predict(6.5)
