### GLOVE CLASSIFICATION IMPORTS ###

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import json
import re
import string

### GLOVE CLASSIFICATION FUNCTIONS ###

# Loads the GloVe Vectors Function:
def load_glove_vectors(glove_file):
  embeddings_index = {}
  with open(glove_file, 'r', encoding='utf-8') as file:
    for line in file:
      values = line.split()
      word = values[0]
      coefs = np.asarray(values[1:], dtype='float32')
      embeddings_index[word] = coefs
  return embeddings_index

# Gets the Average of the Embeddings Vectors:
def get_average_glove_vector(sentence, embeddings_index, embedding_dim):
  words = sentence.split()
  vectors = [embeddings_index.get(word, np.zeros((embedding_dim,))) for word in words]
  if vectors:
    return np.mean(vectors, axis=0)
  else:
    return np.zeros((embedding_dim,))

### GLOVE CLASSIFICATION MODEL ###

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
  text = text.translate(str.maketrans('','', string.punctuation))
  texts.append(text)

  label = data[i]["labels_str"][0]
  labels.append(label)

# Splits the Training and Test Data:
training, testing, y_train, y_test = train_test_split(texts, labels, test_size=0.2)

# Loads the Vectors from File:
glove_file_path = 'glove.6B.25d.txt'
glove_embeddings_index = load_glove_vectors(glove_file_path)

# Gets the Average of the Vectors and Transforms Data:
embedding_dim = len(next(iter(glove_embeddings_index.values())))
x_training = np.array([get_average_glove_vector(sentence, glove_embeddings_index, embedding_dim) for sentence in training])
x_testing = np.array([get_average_glove_vector(sentence, glove_embeddings_index, embedding_dim) for sentence in testing])

# Normalize Shift Data:
x_training_shifted = x_training - np.min(x_training)
x_testing_shifted = x_testing - np.min(x_testing)

# Fits the Multinomial Naive Bayes Model to Training Data:
NBmodel = MultinomialNB()
NBmodel.fit(x_training_shifted, y_train)

# Print Testing and Training Accuracies:
print("Training accuracy: " + str(NBmodel.score(x_training_shifted, y_train)))
print("Testing accuracy: " + str(NBmodel.score(x_testing_shifted, y_test)))