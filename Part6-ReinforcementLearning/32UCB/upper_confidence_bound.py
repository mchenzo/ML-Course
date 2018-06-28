# Upper Confidence Bound

# Importing the libraries
import sys
sys.path.append('/usr/local/lib/python3.6/site-packages')
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('/Users/michaelchen/Desktop/ML-Course/Part6-ReinforcementLearning/32UCB/Ads_CTR_Optimisation.csv')

# Implementing UCB
import math
N = 10000
d = 10
adsSelected = []
numbersOfSelections = [0] * d   # creates a vector of d zeros representing the number of times each ad is selected
sumOfRewards = [0] * d          # vector representing reward from each ad
totalReward = 0
for n in range(0, N):           # this loop is the iterator
    ad = 0
    maxUpperBound = 0
    for i in range(0, d):                       # this loop decides which of the 10 ads to choose each round
        if (numbersOfSelections[i] > 0):         # the contents of this if loop are executed after each ad is picked once; starter info
            averageReward = sumOfRewards[i] / numbersOfSelections[i]
            deltaI = math.sqrt(3/2 * math.log(n + 1) / numbersOfSelections[i])
            upperBound = averageReward + deltaI
        else :
            upperBound = 1e400
        if (upperBound > maxUpperBound): 
            maxUpperBound = upperBound
            ad = i
    adsSelected.append(ad)              # selected ad history vector
    numbersOfSelections[ad] = numbersOfSelections[ad] + 1
    sumOfRewards[ad] = sumOfRewards[ad] + dataset.values[n][ad]
    totalReward = totalReward + dataset.values[n][ad]

# Visualising the results
plt.hist(adsSelected)
plt.title('Histogram of Ad Selections')
plt.xlabel('ads')
plt.ylabel('Number of Times Selected')
plt.show()