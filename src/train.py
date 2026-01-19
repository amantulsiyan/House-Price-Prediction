
#Imports
from sklearn.model_selection import train_test_split
from data import split_features, load_raw_data

# Train/Validation/Test Split
X,y=split_features(load_raw_data())
def split_data(X,y,test_size=0.2,val_size=0.2,random_state=42):
    '''
    Docstring for split_data
    X,y: Split Features.
    test_size: Size of test dataset as a percentage of whole dataset(Should be between 0 and 1)
    val_size: Size of validation dataset as a percentage of remaining dataset(Should be between 0 and 1)
    Sum of test_size + val_size should be less than 1 
    '''
    X_train_full,X_test,y_train_full,y_test=train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )
    X_train,X_valid,y_train,y_valid=train_test_split(
        X_train_full,
        y_train_full,
        val_size=val_size,
        random_state=random_state
    )
    return (X_train,X_valid,X_test,y_train,y_valid,y_test)