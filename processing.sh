#!/bin/bash

# Glove Analysis:
echo "GloVe Analysis:"
echo "Started."
python glove.py
echo "Duration: $SECONDS seconds"

# Formatting:
echo -e '\n\n'
SECONDS=0

# Word2Vec Analysis:
echo "Word2Vec Analysis:"
echo "Started."
python glove.py
echo "Duration: $SECONDS seconds"