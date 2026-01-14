
#Imports
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from model import baseline
from train import X_train
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