### WORD2VEC CLASSIFICATION IMPORTS ###

# Plotting and Standard Imports:
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# Vectorization and Naive Bayes Imports:
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
import json
import re
import string

### WORD2VEC CLASSIFICATION MODEL ###

# Opens the Tweet Dataset:
f = open('MMHS150K_GT.json')

# Loads Data Into Lists and Dictionaries:
data = json.load(f)
texts = []
labels = []

# Gets Labels and Text Data:
f.close()
for i in data:
  text = data[i]["tweet_text"].split("https://")[0]
  text = text.lower()
  mentions = r'@[^ ]'
  text = re.sub(mentions, '', text)
  text = text.translate(str.maketrans('','',string.punctuation))
  texts.append(text)

  label = data[i]["labels_str"][0]
  labels.append(label)

# Splits the Training and Test Data:
training, testing, y_train, y_test = train_test_split(texts, labels, test_size=0.2)

# Vectorizes and Gets the Training Data Features:
featurizer = CountVectorizer()
x_training = featurizer.fit_transform(training)
x_testing = featurizer.transform(testing)

# Fits the Multinomial Naive Bayes Model to Training Data:
NBmodel = MultinomialNB(class_prior=[1, 1, .8, 1, 1, 1])
NBmodel.fit(x_training, y_train)

# Print Testing and Training Accuracies:
print("Training accuracy: " + str(NBmodel.score(x_training, y_train)))
print("Testing accuracy: " + str(NBmodel.score(x_testing, y_test)))