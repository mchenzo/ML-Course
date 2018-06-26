# BACKWARD ELIMINATION
# fit model with all predictors
# remove the predictor with highest p-value
# refit without that variable

# FORWARD SELECTION
# fit all linear regression models, find lowest p val
# refit lowest p val with 1 additional var, find lowest p val combo
# repeat until combined p-value is no longer below threshold

# BIDIRECTIONAL ELIMINATION
# perform 1 step of forward selection
# perform all steps of backward elimination
# repeat until no vars can be added or removed


# Multiple Linear Regression

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
dataset = pd.read_csv('/Users/michaelchen/Desktop/MachineLearningA-ZTemplateFolder/Part2-Regression/5MultipleLinearRegression/50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap; remove first column
# this step is unnecessary, lib will handle it for you
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
# lib handles feature scaling for us
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train) # fit the regressor to the training set

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Building a Backward Elimination Model
import statsmodels.formula.api as sm
# add a col of 1's to factor in the b0 constant
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
# matrix that will only contain variables with high impact on profit
X_opt = X[:, [0,1,2,3,4,5,]] #initially contains all variables
# endog = dependent, exog = number of vars
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 3, 4, 5, ]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 4, 5, ]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 5, ]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, ]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()





# Automated Backwards Elmination
def backwardElim(x, sigLevel):
    # number of columns
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(endog = y, exog = x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)

        if (maxVar > sigLevel):
            for j in range(0, numVars - i):
                if(regressor_OLS.pvalues[j].astype(float) == maxVar):
                    # delete the variable who's p val matches the max pval that exceeded the s.l.
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x

SL = 0.05
X_opt = X
X_Modeled = backwardElim(X_opt, SL)