#!/bin/bash

# Duration Setting:
trap 'echo "Duration: $SECONDS seconds"; exit 1' SIGINT
cd src

# Glove Analysis:
echo "GloVe Analysis:"
echo "Started."
python -B glove.py
echo "Duration: $SECONDS seconds"

# Formatting:
echo -e '\n'
SECONDS=0

# Word2Vec Analysis:
echo "Word2Vec Analysis:"
echo "Started."
python -B word2vec.py
echo "Duration: $SECONDS seconds"

# Formatting:
echo -e '\n'
SECONDS=0

# BERT Analysis:
echo "BERT Analysis:"
echo "Started."
python -B bert.py
echo "Duration: $SECONDS seconds"