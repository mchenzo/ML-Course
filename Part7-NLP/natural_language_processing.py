# Natural Language Processing

# Importing the libraries
import sys
sys.path.append('/usr/local/lib/python3.6/site-packages')
import pandas as pd

# Importing the dataset, use the tsv file because the delimiter is new line not comma
dataset = pd.read_csv('/Users/michaelchen/Desktop/ML-Course/Part7-NLP/Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts; removing numbers, punctuation, reducing different tenses to root word (loved --> love)
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = [] # a collection of same-type texts is called a corpus
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])                             # replace all non-alpha w/ spaces
    review = review.lower()                                                             # reduce to lower case
    review = review.split()                                                             # split into words
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]   # include the word in review if it is not in the stopwords list, stemming
    review = ' '.join(review)
    corpus.append(review)           # append the cleaned and stemmed text

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) #only keep the 1500 most frequent; streamline model
# create a sparse matrix where the rows are each review and the cols are individual stopwords
# the value of each cell is an int indicating the number of times that word showed up in the review
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1] # whether the review was positive or negative; dependent var

# Splitting the dataset into the Training set and Test set


# Fitting Naive Bayes to the Training set


# Predicting the Test set results


# Making the Confusion Matrix

