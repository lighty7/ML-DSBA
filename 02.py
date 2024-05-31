import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import Index

import warnings
warnings.simplefilter("ignore")
df=pd.read_csv(r'./iris.csv')
df.info()
df.isnull().sum()
df.describe()

df.columns
Index(['Id', 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm',
'Species'],
dtype='object')

print(df['variety'].value_counts())