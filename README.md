# 🏠 House Price Prediction (Regression & Feature Engineering)

## 📌 Project Overview
This project focuses on predicting median house prices using the **California Housing dataset**.  
The goal was to build a **robust regression pipeline**, compare baseline and regularized models, and analyze model performance using proper evaluation and explainability techniques.

The project emphasizes:
- Correct train/validation/test discipline
- Feature preprocessing with pipelines
- Model comparison and regularization
- Interpretability through error analysis and feature coefficients

---

## 📊 Dataset
- **Source:** `sklearn.datasets.fetch_california_housing`
- **Samples:** ~20,000
- **Features (8):**
  - Median Income
  - House Age
  - Average Rooms
  - Average Bedrooms
  - Population
  - Average Occupancy
  - Latitude
  - Longitude
- **Target:** Median House Value

All features are numeric; no categorical variables are present.

---

## 🧠 Modeling Approach

### 1. Data Splitting
- Train / Validation / Test split
- Test set held out and used **only once** for final evaluation

### 2. Baseline Model
- **Linear Regression** without preprocessing  
- Used to establish a performance baseline

### 3. Preprocessing Pipeline
Implemented using `scikit-learn` Pipelines:
- Missing value handling: **Mean Imputation**
- Feature scaling: **StandardScaler**
- Combined via **ColumnTransformer**

This ensured no data leakage and consistent preprocessing across models.

---

## 📈 Models Trained
- Linear Regression (baseline & pipeline-based)
- Ridge Regression (L2 regularization)
- Lasso Regression (L1 regularization)

Hyperparameters for Ridge and Lasso were tuned using **5-fold GridSearchCV**.

---

## 📏 Evaluation Metrics
Primary metrics:
- **RMSE (Root Mean Squared Error)**
- **MAE (Mean Absolute Error)**

Metrics were evaluated on:
- Training set (for overfitting check)
- Validation set (for model selection)
- Test set (final, one-time evaluation)

---

## ✅ Results

### Validation Performance
| Model            | RMSE  | MAE  |
|------------------|-------|------|
| Baseline Linear  | 0.728 | 0.533 |
| Pipeline Linear  | 0.728 | 0.533 |
| Ridge Regression | 0.728 | 0.534 |
| Lasso Regression | 0.826 | 0.626 |

- **Selected Model:** Pipeline Linear Regression  
- Regularization did not significantly improve over the scaled linear model

### Generalization Check
- Train RMSE: **0.717**
- Validation RMSE: **0.728**

The small gap indicates **good generalization** and minimal overfitting.

### Test Performance
- **Test RMSE:** 0.750  
- **Test MAE:** 0.533  

---

## 🔍 Model Explainability

### Feature Importance (Linear Coefficients)
Top contributing features by absolute coefficient magnitude:
1. Latitude  
2. Longitude  
3. Median Income  
4. Average Bedrooms  
5. Average Rooms  

This indicates that **location and income** are the strongest drivers of house prices in this dataset.

---

## 📉 Diagnostic Plots
- **Predicted vs Actual scatter plot** to assess bias and variance
- **Residual histogram** to analyze error distribution and consistency

These plots helped verify that errors are reasonably centered and without extreme skew.

---

## 🧩 Key Takeaways
- Feature scaling is essential for linear models
- Regularization did not outperform a well-scaled linear baseline for this dataset
- Location and income dominate price prediction
- Proper evaluation discipline (train vs validation vs test) is critical for trustworthy results

---

## 🛠️ Tech Stack
- Python
- scikit-learn
- pandas
- matplotlib

---

## 🚀 Future Improvements
- Try non-linear models (Random Forest, Gradient Boosting)
- Add spatial feature interactions
- Perform cross-validated learning curve analysis

---

## 📂 How to Run
```bash
python house_price_prediction.py
