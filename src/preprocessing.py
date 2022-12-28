from sklearn.preprocessing import LabelEncoder
from load import load_data, train_path, test_path
import pandas as pd
from featureengineering import time_based_feature_Engineering

def convert_duration_to_minutes(time):
    '''
    This function converts duration in h m to minutes:
    input : hh:mm ,hh, mm
    return:
        min
    '''
    if len(time.split(' ')) >1 :
        hh,mm = time.split(' ')
        hh,mm = int(hh[:-1]),int(mm[:-1])
        duration = hh*60+mm
    else:
        if 'h' in time:
            duration = int(time[:-1])*60
        else:
            duration= int(time[:-1])
            
    return duration


def create_preprocess_date_time(df):
    '''
    This Function preprocess date_of_journey and duration to create departure and arrival date time.
    '''
    df['DepartureDateTime'] = df['Date_of_Journey'] + " "+ df['Dep_Time']
    df['DepartureDateTime'] = pd.to_datetime(df['DepartureDateTime'],infer_datetime_format=True)
    df['Duration_min'] = df['Duration'].apply(lambda x: convert_duration_to_minutes(x))
    df['Duration_timedelta'] = pd.to_timedelta(df['Duration_min'], unit='m')
    df["ArrivalDateTime"] = df['DepartureDateTime'] + df['Duration_timedelta']
    return df

stops_dict = {
    'non-stop':0,
    '2 stops':2,
    '1 stop':1,
    '3 stops':3,
    '4 stops':4
}

## Running the process of prediction is also referred as inference:
def sanity_check(df, train=True):
    '''
    This function performs sanity check on the airline data.
    inputs:
        df: dataframe that we need to perform sanity check
        train: This is used for process of training and inference.
            train is having default value of True and can be set as False if we are running inference.
            
        ## process for training    
        sanity_check(df)
        # process of prediction
        sanity_check(df,train=False)
    returns:
        df
    '''
    if train:
        df.drop_duplicates(inplace=True)
    
    create_preprocess_date_time(df)
    df['Total_Stops'] = df['Total_Stops'].replace(stops_dict)
    df.drop(columns=['Date_of_Journey','Dep_Time','Arrival_Time','Duration','Additional_Info'],axis=1,inplace=True)
    
    return df


def handle_missing_value(df, train=True):
    """
    This function helps to handle missing value.
    Since for Airline data there is just one missing value we can choose to drop missing value.
    inputs:
         df: dataframe which requires imputation.
         
    returns:
        df
    
    """
    df.dropna(inplace=True)
    return df


def frequency_encoder(df, col):
    """
    This function encodes a categorical column based on the frequency of their occurence.
    input:
        df : Input DataFrame in which encoding has to be created 
        col : Column name which has to be encoded
    return: 
          frequency encoded dictionary for columns
    """
    freq_value = df.groupby(col).size()/len(df)
    freq_dict = freq_value.to_dict()
    df["Freq_encoded_"+col] = df[col].replace(freq_dict)
    return freq_dict


def mean_encoder(df, col, target_col):
    """
    This function encodes a categorical column based on the frequency of their occurence.
    input:
        df : Input DataFrame in which encoding has to be created 
        col : Column name which has to be encoded
    return: 
          Mean encoded dict for column
    """
    mean_value = df.groupby(col)[target_col].mean()
    mean_dict = mean_value.to_dict()
    df["Mean_encoded_"+col] = df[col].replace(mean_dict)
    return mean_dict


def label_encoder(df, col):
    """
    This function encodes a categorical column based on the basis of their order label.
    input:
        df : Input DataFrame in which encoding has to be created 
        col : Column name which has to be encoded
    return: 
          label encoded dict for column
    """
    le = LabelEncoder()
    le.fit(df[col])
    label_dict = dict(zip((le.classes_),le.transform(le.classes_)))
    df["Label_encoded_"+col] = df[col].replace(label_dict)
    return label_dict

## Create a function to handle categorical value
def handle_categorical_values(df, target):
    '''
      This function handles categorical value and create a dataframe.
      Input:
        df : Dataframe which require categorical value treatment
      returns :
         Dataframe with all categorical value handled.
    '''
    encoded_dict = {}
    # Getting all object columns
    object_columns = df.select_dtypes(object).columns

    ## generate frequency encoded categorical values
    frequency_encoded_dict ={} 
    for col in object_columns:
        freq_dict = frequency_encoder(df,col)
        frequency_encoded_dict[col] = freq_dict

    ## generate target mean encoded categorical values
    mean_encoded_dict ={} 
    for col in object_columns:
        mean_dict = mean_encoder(df,col,target)
        mean_encoded_dict[col] = mean_dict

        ## generate label encoded categorical values
    label_encoded_dict ={} 
    for col in object_columns:
        label_dict = label_encoder(df,col)
        label_encoded_dict[col] = label_dict
    
    encoded_dict["Frequency"] = frequency_encoded_dict
    encoded_dict["Mean"] = mean_encoded_dict
    encoded_dict["Label"] = label_encoded_dict
    #print(encoded_dict)
    return df, encoded_dict


def airline_handle_categorical_data(df, target):
    df['Destination'] = df['Destination'].replace({'New Delhi':'Delhi'})
    df, encoded_dict = handle_categorical_values(df, target)
    #print(encoded_dict)
    categorical_cols = df.select_dtypes(object).columns
    df.drop(columns=categorical_cols,inplace=True)
    return df, encoded_dict


if __name__ == "__main__":
    # test your module here
    target = 'Price'
    df = load_data(train_path)
    df = sanity_check(df, train=True)
    handle_missing_value(df)
    df, encoded_dict = airline_handle_categorical_data(df, target)
    final_df = time_based_feature_Engineering(df)
    print(df.info())
    print(encoded_dict)
    print(final_df.info())