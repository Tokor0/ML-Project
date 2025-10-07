from pickle import load
import numpy as np
from os.path import join


# All my homies hate boilerplate
def load_from_fname(fname):
    with open(join("models", fname), "rb") as f:
        return load(f)


models = [
    (name, load_from_fname(fname))
    for name, fname in [
        ("Logistic regression", "logReg-model.pkl"),
        ("Random forest", "randForest-model.pkl"),
    ]
]

while True:
    msg = input("Input message:\n")
    for name, model in models:
        print(f"{name}: {model.predict([msg])}")
