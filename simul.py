
# Imports 

from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    root_mean_squared_error,
    mean_squared_error,
    mean_absolute_error
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV

# Load dataset directly as a pandas DataFrame & EDA

data = fetch_california_housing(as_frame=True)
df = data.frame
X=df.drop('MedHouseVal',axis=1)
y=df['MedHouseVal']

# Train/Validation/Test Split

X_train_full,X_test,y_train_full,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
X_train,X_valid,y_train,y_valid=train_test_split(X_train_full,y_train_full,train_size=0.75,random_state=42)

# Baseline Model: Linear Regression

baseline=LinearRegression()
baseline.fit(X_train,y_train)
val_preds=baseline.predict(X_valid)
baseline_val_score_rmse=root_mean_squared_error(y_valid,val_preds)
baseline_val_score_mae=mean_absolute_error(y_valid,val_preds)
print("Predicted Score on Validation Set in Baseline Model:",val_preds)
print("RMSE Score in Baseline Model:",baseline_val_score_rmse)
print("MAE Score in Baseline Model:",baseline_val_score_mae)

# Feature Engineering + Proper Pipeline

numeric_features=X_train.columns.tolist()
categorical_features=[]
numeric_pipeline=Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='mean')),
    ('scaler',StandardScaler())    
])
categorical_pipeline=Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('onehot',OneHotEncoder(handle_unknown='ignore'))
])
preprocessor=ColumnTransformer(
    transformers=[
        ('num', numeric_pipeline,numeric_features),
        ('cat',categorical_pipeline,categorical_features)
    ]
)
pipeline=Pipeline(steps=[
    ('preprocessing',preprocessor),
    ('regressor',LinearRegression())
])
pipeline.fit(X_train,y_train)
val_preds_pipeline=pipeline.predict(X_valid)
pipeline_val_score_rmse=root_mean_squared_error(y_valid,val_preds_pipeline)
pipeline_val_score_mae=mean_absolute_error(y_valid,val_preds_pipeline)
test_preds_pipeline=pipeline.predict(X_test)
print("Predicted Score on Validation Set after Pipelining:",val_preds_pipeline)
print("RMSE Score after Pipelining:",pipeline_val_score_rmse)
print("MAE Score after Pipelining:",pipeline_val_score_mae)
print("Predicted Score on Test Set after Pipelining:",test_preds_pipeline)

# Ridge 

ridge_model=Pipeline(steps=[
    ('preprocessing',preprocessor),
    ('ridge',Ridge(alpha=1.0))
])
ridge_param_grid={
    'ridge__alpha':[0.001,0.01,0.1,1,10,100]
}
ridge_grid=GridSearchCV(
    estimator=ridge_model,
    param_grid=ridge_param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=1
)
ridge_grid.fit(X_train,y_train)
best_ridge_model=ridge_grid.best_estimator_
best_ridge_alpha=ridge_grid.best_params_['ridge__alpha']
val_preds_ridge=best_ridge_model.predict(X_valid)
ridge_val_score_rmse=root_mean_squared_error(y_valid,val_preds_ridge)
ridge_val_score_mae=mean_absolute_error(y_valid,val_preds_ridge)
print("Predicted Score on Validation Set after Ridge Regularisation:",val_preds_ridge)
print("RMSE Score after Ridge Regularisation:",ridge_val_score_rmse)
print("MAE Score after Ridge Regularisation:",ridge_val_score_mae)

# Lasso

lasso_model=Pipeline(steps=[
    ('preprocessing',preprocessor),
    ('lasso',Lasso(alpha=0.01))
])
lasso_param_grid={
    'lasso__alpha':[0.001,0.01,0.1,1,10,100]
}
lasso_grid=GridSearchCV(
    estimator=lasso_model,
    param_grid=lasso_param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=1
)
lasso_grid.fit(X_train,y_train)
best_lasso_model=lasso_grid.best_estimator_
best_lasso_alpha=lasso_grid.best_params_['lasso__alpha']
val_preds_lasso=best_lasso_model.predict(X_valid)
lasso_val_score_rmse=root_mean_squared_error(y_valid,val_preds_lasso)
lasso_val_score_mae=mean_absolute_error(y_valid,val_preds_lasso)
print("Predicted Score on Validation Set after Lasso Regularisation:",val_preds_lasso)
print("RMSE Score after Lasso Regularisation:",lasso_val_score_rmse)
print("MAE Score after Lasso Regularisation:",lasso_val_score_mae)