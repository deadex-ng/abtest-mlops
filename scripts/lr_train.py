import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn 
#from urlparse import urlparse
from urllib.parse import urlparse
import warnings
import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2
    
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    np.random.seed(40)

    # Read the AdsmartABdata csv file from the URL
    csv_url = (
        "https://raw.githubusercontent.com/deadex-ng/smartAd_ab_testing/main/data/AdSmartABdata.csv"
    )
    try:
        df = pd.read_csv(csv_url)
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )  
 
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
        logreg = LogisticRegression()
        logreg.fit(x_train, y_train)

        y_prediction = logreg.predict(x_val)
        #use validatation date set to calculate loss function
        (rmse, mae, r2) = eval_metrics(y_val,y_prediction)
        
        #loss functions for this model are: mean absolute error(mae) and root mean squared error(rmse)
        #r2 is the r2_score
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(logreg, "model", registered_model_name="LogisticRegressionAdmodel")
        else:
            mlflow.sklearn.log_model(logreg, "model")
