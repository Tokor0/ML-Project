import os
from pickle import dump
from tabulate import tabulate

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_validate

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
<<<<<<< HEAD
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
=======
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
)
>>>>>>> bcc428d (Added random forest classifier)
from sklearn.pipeline import Pipeline

DATA_PATH = os.path.join("balanced_sentiment_dataset.csv")
PLOT_DIR = "plots"
MODEL_DIR = "models"
CM_PLOT_FNAME = "-cm.png"
ROC_PLOT_FNAME = "-roc.png"
MODEL_FNAME = "-model.pkl"


def cm_plot(y_true, y_pred, mname):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(f"{mname}-cm")
    sns.heatmap(cm, annot=True)
    plt.title("Confusion Matrix")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.savefig(os.path.join(PLOT_DIR, mname + CM_PLOT_FNAME))


def roc_plot(y_true, y_score, mname):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(f"{mname}-roc")
    plt.plot([0, 1], [0, 1], linestyle=":", color="gray")
    plt.plot(fpr, tpr)
    plt.title("ROC Curve")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.savefig(os.path.join(PLOT_DIR, mname + ROC_PLOT_FNAME))


def dump_model(mname):
    with open(os.path.join(MODEL_DIR, mname + MODEL_FNAME), "wb") as f:
        dump(model_pipeline, f, protocol=5)


# Function for producing a tabulable structure
# from the return type of cross_validate.
def mean_std_stats(cv_res, *args):
    res = []
    for a in args:
        res.append([f"Mean {a[0]}", np.mean(cv_res[a[1]])])
        res.append([f"StD {a[0]}", np.std(cv_res[a[1]])])
    return res


# Logging function
def log(msg, f, *args, **kwargs):
    print(msg, end=" ", flush=True)
    res = f(*args, **kwargs)
    print("DONE!")
    return res


data = pd.read_csv(DATA_PATH)

# Assign the feature and label vectors

X = data["text"]
y = data["sentiment"]

# Split the data to training and testing subsets.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

vectorizer = TfidfVectorizer(
    stop_words="english", max_features=5000, token_pattern="\\w+|[^\\w\\s]"
)

<<<<<<< HEAD
model1 = LogisticRegression(max_iter=1000)
model_pipeline1 = Pipeline([
    ('transformer', vectorizer),
    ('classifier', model1)
])
=======
models = [
    ("logReg", LogisticRegression(max_iter=1000)),
    ("randForest", RandomForestClassifier(criterion="log_loss")),
]
model_pipelines = [
    (model[0], Pipeline([("transformer", vectorizer), ("classifier", model[1])]))
    for model in models
]
# model_pipeline = Pipeline([("transformer", vectorizer), ("classifier", model)])
>>>>>>> bcc428d (Added random forest classifier)

model2 = RandomForestClassifier(n_estimators=100)
model_pipeline2 = Pipeline([
    ('transformer', vectorizer),
    ('classifier', model2)
])

model3 = SVC(max_iter=1000)
model_pipeline3 = Pipeline([
    ('transformer', vectorizer),
    ('classifier', model3)
])

model4 = MLPClassifier(hidden_layer_sizes=(100,))
model_pipeline4 = Pipeline([
    ('transformer', vectorizer),
    ('classifier', model4)
])

model=model4
model_pipeline=model_pipeline4


dims = [
    ["Total dim.", X.shape],
    ["Train dim.", X_train.shape],
]

print(tabulate(dims))

for m in model_pipelines:

    model_name = m[0]
    model_pipeline = m[1]

    print(model_name.upper() + "\n")
    log("Fitting...", model_pipeline.fit, X_train, y_train)

    y_pred = model_pipeline.predict(X_test)
    y_score = model_pipeline.predict_proba(X_test)[:, 1]
    metrics = [
        ["Accuracy", accuracy_score(y_test, y_pred)],
        ["Precision", precision_score(y_test, y_pred)],
        ["ROC AUC", roc_auc_score(y_test, y_pred)],
    ]
    print(tabulate(metrics))

    log(
        f"Saving confusion matrix plot...",
        cm_plot,
        y_test,
        y_pred,
        model_name,
    )

    log(
        f"Saving ROC plot...",
        roc_plot,
        y_test,
        y_score,
        model_name,
    )

    log(f"Saving model...", dump_model, model_name)

    # K-Fold cross validation

    cv_res = log(
        "Cross validating...",
        cross_validate,
        estimator=model_pipeline,
        X=X,
        y=y,
        cv=5,
        n_jobs=5,
        scoring=["accuracy", "precision"],
    )

    stats = mean_std_stats(
        cv_res,
        ("Fit t", "fit_time"),
        ("Score t", "score_time"),
        ("acc.", "test_accuracy"),
        ("prec.", "test_precision"),
    )

    print(tabulate(stats))
