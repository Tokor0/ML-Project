import os
import re
from pickle import dump

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn.pipeline import Pipeline

datapath = os.path.join('balanced_sentiment_dataset.csv')
data = pd.read_csv(datapath)

print("Data dim.:")
print(data.shape)

vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000,
    token_pattern='\\w+|[^\\w\\s]'
)

X = data['text']
y = data['sentiment']

# Split the data to training and testing subsets.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

print("Training data dim.:")
print(X_train.shape)

model = LogisticRegression(max_iter=1000)
model_pipeline = Pipeline([
    ('transformer', vectorizer),
    ('classifier', model)
])

print("Fitting...")

model_pipeline.fit(X_train, y_train)

print("DONE!\n")

y_pred = model_pipeline.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)

print(f"Accuracy: {acc}")
print(f"Precision: {prec}")

print("Saving model to model.pkl...")

with open("model.pkl", "wb") as f:
    dump(model_pipeline, f, protocol=5)

print("DONE!\n")

# K-Fold cross validation

print("Cross validating...")

cv_res = cross_validate(
    estimator=model_pipeline,
    X=X,
    y=y,
    cv=5,
    scoring=['accuracy', 'precision'],
)

print("DONE!\n")

print(cv_res)


