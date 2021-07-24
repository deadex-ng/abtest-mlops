import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from numpy import mean
from numpy import std
import mlflow
import mlflow.sklearn 
#from urlparse import urlparse
from urllib.parse import urlparse
import warnings
import logging
import dvc.api

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

    
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    np.random.seed(40)

    #read data from dvc remote storarge 
    path = '/home/ubuntu/Desktop/project_1/mlops4/abtest-mlops/data/Ab_data.csv'
    repo = '/home/ubuntu/Desktop/project_1/mlops4/abtest-mlops'
    version = 'v5'


    data_url = dvc.api.get_url(
        path =path,
        repo = repo,
        rev = version
        )

    df = pd.read_csv(data_url, sep = ",")  
 
    """Transform categorical data using LabelEncoder"""
    
    experiment = LabelEncoder()
    df ['experiment'] = experiment.fit_transform(df['experiment'].astype('str'))
    df ['device_make'] = experiment.fit_transform(df['device_make'].astype('str'))
    df ['browser'] = experiment.fit_transform(df['browser'].astype('str'))


    #Extract day from data
    #Discard year and month because the experiment was done on one month(July) and one year(2020)
    df['datetime'] = pd.to_datetime(df['date'],format='%Y-%m-%d')
    df.drop('date',axis=1,inplace=True)
    df['day'] = df['datetime'].dt.strftime('%d')
    df.drop('datetime',axis=1,inplace=True)


    #all values are numeric except auction_id
    df[['day']] = df [['day']].apply(pd.to_numeric)

    #drop id since it won't be needed for our model
    df.drop('auction_id',axis=1,inplace=True)

    #separate the other attributes from the predicting attribute
    x = df.drop('yes',axis=1)
    #separte the predicting attribute into y for model training 
    y = df['yes'].values.reshape(-1,1)

    #set the ratio for splitting the data
    train_ratio = 0.70
    validation_ratio =0.20
    test_ratio = 0.10

    #train is now 70% of the entires dataset
    x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=1 - train_ratio)

    #test is now 10% of the initial data set 
    #validation is now 20% of the initial data set
    x_val,x_test,y_val,y_test =train_test_split(x_test,y_test,test_size= test_ratio/(test_ratio + validation_ratio))

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5



    with mlflow.start_run():
        rfe = RFE(estimator=DecisionTreeRegressor(), n_features_to_select=2)

        model = XGBClassifier()
        pipeline = Pipeline(steps=[('s',rfe),('m',model)])

        cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
        n_scores = cross_val_score(pipeline, x_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
        # report performance
        print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
        
        mlflow.log_metric("mae",mean(n_scores))
        mlflow.log_metric("std",std(n_scores))

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(model, "model", registered_model_name="XgBoostAdmodel")
        else:
            mlflow.sklearn.log_model(model, "model")
