# Simple Linear Regression

# Importing the dataset
dataset = read.csv('Salary_Data.csv')

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)

# 20 observations in the training set, 20 in test set
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting simple linear regression to the training set
# @param formula (salary proportional to yrs exp)
# @param data (training set to derive fit)
regressor = lm(formula = Salary ~ YearsExperience, 
               data = training_set)

# Predicting Test set results
y_pred = predict(regressor, newdata = test_set)

# Visualizing Training set results
# install.packages('ggplot2')
# automatically include ggplot2 library
library(ggplot2)

ggplot() + 
  # plots the points from the training set w/ respective axes
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
            color = 'red') +
  # plot the regression line
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            color = 'blue') +
  ggtitle('Salary vs. Experience (Training Set)') +
  # Label x axis
  xlab('Years of Experience') +
  ylab('Salary')


# Visualizing Test set results
ggplot() + 
  # plots the points from the test set w/ respective axes 
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
            color = 'red') +
  # plot the same regression line regression line
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            color = 'blue') +
  ggtitle('Salary vs. Experience (Test Set)') +
  # Label x axis
  xlab('Years of Experience') +
  ylab('Salary')
