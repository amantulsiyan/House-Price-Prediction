
# Imports 

import matplotlib.pyplot as plt
from sklearn.metrics import (
    root_mean_squared_error,
    mean_squared_error,
    mean_absolute_error
)

from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV







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