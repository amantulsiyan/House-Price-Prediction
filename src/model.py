
#Imports
from sklearn.linear_model import LinearRegression
from train import (
    X_train, 
    y_train
)

# Baseline Model: Linear Regression
baseline=LinearRegression()
baseline.fit(X_train,y_train)