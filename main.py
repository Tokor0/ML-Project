import os
import re
from pickle import dump
from tabulate import tabulate

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn.pipeline import Pipeline

DATA_PATH = os.path.join('balanced_sentiment_dataset.csv')
PLOT_DIR = "plots"
CM_PLOT_FNAME = os.path.join(PLOT_DIR, "cm.png")
MODEL_FNAME = "model.pkl"

def cm_plot(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True)
    plt.title("Confusion Matrix")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.savefig(CM_PLOT_FNAME)

def dump_model(fname):
    with open(fname, "wb") as f:
        dump(model_pipeline, f, protocol=5)

# Logging function
def log(msg, f, *args, **kwargs):
    print(msg, end=' ', flush=True)
    res = f(*args, **kwargs)
    print("DONE!")
    return res

data = pd.read_csv(DATA_PATH)

# Assign the feature and label vectors

X = data['text']
y = data['sentiment']

# Split the data to training and testing subsets.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000,
    token_pattern='\\w+|[^\\w\\s]'
)

model = LogisticRegression(max_iter=1000)
model_pipeline = Pipeline([
    ('transformer', vectorizer),
    ('classifier', model)
])

dims = [
        ["Total dim.", X.shape],
        ["Train dim.", X_train.shape],
]

print(tabulate(dims))

log("Fitting...", model_pipeline.fit, X_train, y_train)

y_pred = model_pipeline.predict(X_test)
metrics = [
        ["Accuracy", accuracy_score(y_test, y_pred)],
        ["Precision", precision_score(y_test, y_pred)]
]
print(tabulate(metrics))

log(
    f"Saving confusion matrix plot to {CM_PLOT_FNAME}...",
    cm_plot, y_test, y_pred
)

log(
    f"Saving model to {MODEL_FNAME}...",
    dump_model,
    MODEL_FNAME
)

# K-Fold cross validation

cv_res = log(
    "Cross validating...",
    cross_validate,
    estimator=model_pipeline,
    X=X,
    y=y,
    cv=5,
    scoring=['accuracy', 'precision'],
)

print(cv_res)


