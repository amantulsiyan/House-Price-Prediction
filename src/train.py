
#Imports
from sklearn.model_selection import train_test_split
from data import X,y

# Train/Validation/Test Split

X_train_full,X_test,y_train_full,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
X_train,X_valid,y_train,y_valid=train_test_split(X_train_full,y_train_full,train_size=0.75,random_state=42)