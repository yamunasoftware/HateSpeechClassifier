### DISTILBERT CLASSIFICATION IMPORTS ###

from transformers import DistilBertTokenizer, DistilBertModel
import torch
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import json
import re
import string

### DISTILBERT CLASSIFICATION FUNCTIONS ###

# Encode Training Data from BERT Embeddings:
def encode_texts(texts):
  tokens = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
  with torch.no_grad():
    outputs = model(**tokens)
  return outputs.last_hidden_state[:, 0, :].numpy()

### DISTILBERT CLASSIFICATION MODEL ###

# Opens the Tweet Dataset:
with open('MMHS150K_GT.json') as f:
  data = json.load(f)

# Loads Data Into Lists:
texts = []
labels = []

# Gets Labels and Text Data:
for i in data:
  text = data[i]["tweet_text"].split("https://")[0]
  text = text.lower()
  mentions = r'@[^ ]'
  text = re.sub(mentions, '', text)
  text = text.translate(str.maketrans('', '', string.punctuation))
  texts.append(text)

  label = data[i]["labels_str"][0]
  labels.append(label)

# Splits the Training and Test Data:
training, testing, y_train, y_test = train_test_split(texts, labels, test_size=0.2)

# Tokenize and Encode Texts Using DistilBERT:
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Generate DistilBERT Embeddings for Training and Testing Data:
x_training = encode_texts(training)
x_testing = encode_texts(testing)

# Convert embeddings to positive values for MultinomialNB:
x_training = np.abs(x_training)
x_testing = np.abs(x_testing)

# Fits the Multinomial Naive Bayes Model to Training Data:
NBmodel = MultinomialNB()
NBmodel.fit(x_training, y_train)

# Print Testing and Training Accuracies:
print("Training accuracy: " + str(NBmodel.score(x_training, y_train)))
print("Testing accuracy: " + str(NBmodel.score(x_testing, y_test)))
