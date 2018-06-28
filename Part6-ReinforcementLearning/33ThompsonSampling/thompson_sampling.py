# Thompson Sampling

# Importing the libraries
import sys
sys.path.append('/usr/local/lib/python3.6/site-packages')
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('/Users/michaelchen/Desktop/ML-Course/Part6-ReinforcementLearning/33ThompsonSampling/Ads_CTR_Optimisation.csv')

# Implementing Thompson Sampling
import random
N = 10000
d = 10
adsSelected = []
numbersOfRewards1 = [0] * d
numbersOfRewards0 = [0] * d
totalReward = 0
for n in range(0, N):           # this loop is the iterator
    ad = 0
    maxRandom = 0
    # this loop decides which of the 10 ads to choose each round
    for i in range(0, d):
        randomBeta = random.betavariate(numbersOfRewards1[i] + 1, numbersOfRewards0[i] + 1)
        if (randomBeta > maxRandom):
            maxRandom = randomBeta
            ad = i
    adsSelected.append(ad)              # selected ad history vector
    if (dataset.values[n][ad] == 1):
        numbersOfRewards1[ad] += 1
    else: 
        numbersOfRewards0[ad] += 1
    totalReward = totalReward + dataset.values[n][ad]

# Visualising the results - Histogram
plt.hist(adsSelected)
plt.title('Histogram of Ad Selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
