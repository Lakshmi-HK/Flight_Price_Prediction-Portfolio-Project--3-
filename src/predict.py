from load import load_data, train_path, test_path
from featureengineering import time_based_feature_Engineering
from preprocessing import sanity_check, handle_missing_value, airline_handle_categorical_data
from model import train, xgboost, load_pickle, save_pickle
from check import list_A

import pandas as pd
import numpy as np 

def encode_predict_input(df, encoded_dict, error_handle=True):
    '''
    This function encodes categorical values with same values as training encoded values.
    Input:
      df : DataFrame
      encoded_dict : Category encoded dictionary
    returns :None
    '''
    encoded_cols = ['Airline', 'Source', 'Destination', 'Route']
    
    frequency_dict = encoded_dict['Frequency']
    mean_dict = encoded_dict['Mean']
    label_dict = encoded_dict['Label']
    for col in encoded_cols:
        df["Freq_encoded_"+ col] = df[col].replace(frequency_dict[col])
        df["Mean_encoded_"+ col] = df[col].replace(mean_dict[col])
        df["Label_encoded_"+ col] = df[col].replace(label_dict[col])
    df.drop(columns=encoded_cols,inplace=True)
    #this is to handle unseen routes in test file.
    # replacing unseen values with -1.
    # df[df['Freq_encoded_Route'].str.contains('→',na=False)].index.to_list() --> [6 , 72, 484 , 3838, 1980]
    
    if error_handle:
        for index in [6 , 72, 484 , 3838, 1980]:
            df.loc[index,'Freq_encoded_Route'] = -1
            df.loc[index,'Mean_encoded_Route'] = -1
            df.loc[index,'Label_encoded_Route'] = -1

        df['Freq_encoded_Route'] =  df['Freq_encoded_Route'].astype('float')
        df['Mean_encoded_Route'] =  df['Mean_encoded_Route'].astype('float')
        df['Label_encoded_Route'] =  df['Label_encoded_Route'].astype('float')
    return df


def predict_price(df, encoded_path, model_path, error_handle=True):
    sanity_check(df, train=False)

    loaded_dict = load_pickle(encoded_path)

    df['Destination'] = df['Destination'].replace({'New Delhi':'Delhi'})
    df = encode_predict_input(df, loaded_dict, error_handle=True)

    test_X = time_based_feature_Engineering(df)

    loaded_model = load_pickle(model_path)
    test_X = test_X[list_A] # this is to handle feature mismatch in xgboost for inference.
    result = loaded_model.predict(test_X)
    return result


model_path = '../models/xgboost_demo.pickle'
encoded_path = '../models/encoded_dict_demo.pickle'

df = load_data(test_path)
df= df.iloc[:,:]


output = predict_price(df, encoded_path, model_path, error_handle=True)
print(output)

output_path = "../output/results.pickle"
save_pickle(output_path, output)