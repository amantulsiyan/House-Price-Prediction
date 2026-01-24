# =========================
# Imports
# =========================
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd

# =========================
# Load Data
# =========================
data = fetch_california_housing(as_frame=True)
df = data.frame

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]
print(X.columns.tolist())

# =========================
# Train / Validation / Test Split
# =========================
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=42
)


# =========================
# Baseline Model
# =========================
baseline_model = LinearRegression()
baseline_model.fit(X_train, y_train)

baseline_val_preds = baseline_model.predict(X_valid)
baseline_rmse = root_mean_squared_error(y_valid, baseline_val_preds)
baseline_mae = mean_absolute_error(y_valid, baseline_val_preds)


# =========================
# Preprocessing Pipeline
# =========================
numeric_features = X_train.columns.tolist()

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features)
    ]
)


# =========================
# Linear Regression Pipeline
# =========================
linear_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]
)

linear_pipeline.fit(X_train, y_train)

linear_val_preds = linear_pipeline.predict(X_valid)
linear_rmse = root_mean_squared_error(y_valid, linear_val_preds)
linear_mae = mean_absolute_error(y_valid, linear_val_preds)


# =========================
# Ridge Regression (GridSearch)
# =========================
ridge_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", Ridge())
    ]
)

ridge_param_grid = {
    "model__alpha": [0.001, 0.01, 0.1, 1, 10, 100]
}

ridge_search = GridSearchCV(
    ridge_pipeline,
    ridge_param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

ridge_search.fit(X_train, y_train)

best_ridge = ridge_search.best_estimator_

ridge_val_preds = best_ridge.predict(X_valid)
ridge_rmse = root_mean_squared_error(y_valid, ridge_val_preds)
ridge_mae = mean_absolute_error(y_valid, ridge_val_preds)


# =========================
# Lasso Regression (GridSearch)
# =========================
lasso_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", Lasso(max_iter=10000))
    ]
)

lasso_param_grid = {
    "model__alpha": [0.001, 0.01, 0.1, 1, 10]
}

lasso_search = GridSearchCV(
    lasso_pipeline,
    lasso_param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

lasso_search.fit(X_train, y_train)

best_lasso = lasso_search.best_estimator_

lasso_val_preds = best_lasso.predict(X_valid)
lasso_rmse = root_mean_squared_error(y_valid, lasso_val_preds)
lasso_mae = mean_absolute_error(y_valid, lasso_val_preds)


# =========================
# Model Comparison (Validation)
# =========================
results = {
    "Baseline Linear": (baseline_rmse, baseline_mae),
    "Pipeline Linear": (linear_rmse, linear_mae),
    "Ridge": (ridge_rmse, ridge_mae),
    "Lasso": (lasso_rmse, lasso_mae),
}

best_model_name = min(results, key=lambda x: results[x][0])

if best_model_name == "Baseline Linear":
    final_model = baseline_model
elif best_model_name == "Pipeline Linear":
    final_model = linear_pipeline
elif best_model_name == "Ridge":
    final_model = best_ridge
else:
    final_model = best_lasso


# =========================
# Final Evaluation on Test Set (ONCE)
# =========================
test_preds = final_model.predict(X_test)
test_rmse = root_mean_squared_error(y_test, test_preds)
test_mae = mean_absolute_error(y_test, test_preds)


# =========================
# Final Output
# =========================
print("Validation Results (RMSE, MAE):")
for model, scores in results.items():
    print(f"{model}: RMSE={scores[0]:.4f}, MAE={scores[1]:.4f}")

print("\nSelected Model:", best_model_name)
print(f"Test RMSE: {test_rmse:.4f}")
print(f"Test MAE: {test_mae:.4f}")

train_preds=final_model.predict(X_train)
train_preds_rmse=root_mean_squared_error(y_train,train_preds)
train_preds_mae=mean_absolute_error(y_train,train_preds)

valid_preds=final_model.predict(X_valid)
valid_preds_rmse=root_mean_squared_error(y_valid,valid_preds)
valid_preds_mae=mean_absolute_error(y_valid,valid_preds)

print("Train RMSE:", train_preds_rmse)
print("Validation RMSE:", valid_preds_rmse)
print("Train MAE:", train_preds_mae)
print("Validation MAE:", valid_preds_mae)

#Actual vs Predicted Plot
plt.figure(figsize=(10,10))
plt.scatter(y_test,test_preds)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Regression Plot")
plt.plot(plt.xlim(), plt.xlim(), 'k--', label="x=y")
plt.legend()
plt.show()

#Residual Histogram
residuals=y_test-test_preds
plt.figure(figsize=(10,10))
plt.hist(residuals, bins=30)
plt.axvline(0,color='red', linestyle='--',label="X=0")
plt.xlabel("Residual Values")
plt.ylabel("Frequencies")
plt.title("Residual Value Histogram")
plt.grid(True)
plt.legend()
plt.show()

feature_names=final_model.named_steps["preprocessor"].get_feature_names_out()
coefficients=final_model.named_steps["model"].coef_
coef_df=pd.DataFrame({
    "feature":feature_names,
    "coefficient":coefficients
})
coef_df['coefficient']=coef_df["coefficient"].abs()
coef_df=coef_df.sort_values(by='coefficient',ascending=False)
print(coef_df)