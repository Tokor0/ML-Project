import os
datapath=os.path.join('superconductivty+data','train.csv')
import numpy as np
import pandas as pd
data=pd.read_csv(datapath)
print(data.head(5))
