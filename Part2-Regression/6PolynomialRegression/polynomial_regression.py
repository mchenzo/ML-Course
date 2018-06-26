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


# Reference Linear Regression Model
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(X, Y)

# Fitting a Polynomial Regression Model
from sklearn.preprocessing import PolynomialFeatures
poly_regressor = PolynomialFeatures(degree = 4)
# fit the regressor to X, then transform X, generates a column of 1's at beginning to account for intercept
X_poly = poly_regressor.fit_transform(X)
lin_regressor_2 = LinearRegression()
lin_regressor_2.fit(X_poly, Y)

# Visualize Linear Regression
plt.scatter(X, Y, color = 'red')
plt.plot(X, linear_regressor.predict(X), color = 'blue')
plt.title('Salary vs. Position Level')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualize Polynomial Regression
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, lin_regressor_2.predict(poly_regressor.fit_transform(X_grid)), color='blue')
plt.title('Salary vs. Position Level (Polynomial')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


# Using Linear Regression to Predict Salaries
linear_regressor.predict(6.5)

# Using Polynomial Regression to Predict Salaries
lin_regressor_2.predict(poly_regressor.fit_transform(6.5))