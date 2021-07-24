import pandas as pd 
import dvc.api
path = '/home/ubuntu/Desktop/project_1/mlops4/abtest-mlops/data/Adsmart_v2.csv'
#path = 'data/Adsmart_v2.csv'
repo = '/home/ubuntu/Desktop/project_1/mlops4/abtest-mlops'
version = 'v2'


data_url = dvc.api.get_url(
    path =path,
    repo = repo,
    rev = version)

data = pd.read_csv(data_url,sep = ",")
print(data.columns.tolist)
