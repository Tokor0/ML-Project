import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

datapath = os.path.join('balanced_sentiment_dataset.csv')
data = pd.read_csv(datapath)

def tokenize(str, max_tokens=100, empty_tok=' '):
    # Get a list of words, and non-whitespace and non-word symbols.
    text_tokens = re.findall(
        r'\w+|[^\w\s]',
        # Some texts have \\n when it should be \n...
        # Look for other things that may need cleaning up.
        str.replace("\\n", "\n")
    )

    diff = max_tokens - len(text_tokens)
    if diff < 0:
        # If too many tokens, return NaN.
        return np.nan

    # Otherwise, return padded token list and n. of
    # non-empty tokens.
    return text_tokens + [empty_tok] * diff

data['tokens'] = data['text'].map(tokenize)

# Drop NaN rows.
data = data.dropna()

print(data['tokens'][0])
print(len(data['tokens'][0]))


