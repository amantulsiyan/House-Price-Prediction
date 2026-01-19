
#Imports

from sklearn.datasets import fetch_california_housing
def load_raw_data():
    '''
    Fetches California Housing Dataset from sklearn and returns a dataframe
    '''
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    return df
def split_features(df):
    '''
    Parameters: Takes dataframe 'df' as input and returns the features into X & y variables
    '''
    X=df.drop('MedHouseVal',axis=1)
    y=df['MedHouseVal']
    return (X,y)