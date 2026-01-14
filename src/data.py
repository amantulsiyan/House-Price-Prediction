
#Imports

from sklearn.datasets import fetch_california_housing
import pandas as pd

data = fetch_california_housing(as_frame=True)
df = data.frame
X=df.drop('MedHouseVal',axis=1)
y=df['MedHouseVal']