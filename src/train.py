from load import load_data, train_path, test_path
from featureengineering import time_based_feature_Engineering
from preprocessing import sanity_check, handle_missing_value, airline_handle_categorical_data
from model import train, xgboost, save_pickle

import pandas as pd
import numpy as np 

target ='Price'

model_path = '../models/xgboost_demo.pickle'
encoded_path = '../models/encoded_dict_demo.pickle'



## process for training   
df = load_data(train_path)
sanity_check(df)
handle_missing_value(df)

df, encoded_dict = airline_handle_categorical_data(df,target)
final_df = time_based_feature_Engineering(df)
X = final_df.drop(columns=target)
y = final_df[target]

print(X.columns)

model = train(X,y,xgboost)

save_pickle(encoded_path,encoded_dict)
save_pickle(model_path,model)