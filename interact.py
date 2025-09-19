from pickle import load
import numpy as np

with open("model.pkl", "rb") as f:
    model = load(f)

while True:
    msg = input("Input message:\n")
    print(model.predict([msg]))
