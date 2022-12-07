import numpy as np
import pandas as  pd

def time_of_day(hr):
    '''
    This function gives the time of day based on logic:
        # 3-8 early_morning or 1
        # 8-12 morning or 2
        # 12-16 afternoon or 3
        # 16-20 evening or 4
        # 20-00 night or 5
        # 00-3 late_night or 6
        # invalid or 0
    input:
        hr
    return: tuple
        (timeOfDay,timeOfDay_encoded
    '''
    if hr in range(0,3) :
        str_val = 'late_night'
        val = 6
    elif hr in range(20,23):
        str_val = 'night'
        val = 5
    elif hr in range(16,20):
        str_val = 'evening'
        val = 4
    elif hr in range(12,26):
        str_val = 'after_noon'
        val = 3
    elif hr in range(8,12):
        str_val = 'morning'
        val = 2
    elif hr in range(3,8):
        str_val = 'early_morning'
        val = 1
    else:
        str_val = 'invalid'
        val = 0
    return (str_val, val)


def time_based_feature_Engineering(df):

    df['dep_hr'] = df['DepartureDateTime'].dt.hour
    df['arr_hr'] = df['ArrivalDateTime'].dt.hour

    df['dep_month'] = df['DepartureDateTime'].dt.month
    df['dep_day_of_month'] = df['DepartureDateTime'].dt.day

    df['arr_month'] = df['ArrivalDateTime'].dt.month
    df['arr_day_of_month'] = df['ArrivalDateTime'].dt.day

    df['dep_day_of_week'] = df['DepartureDateTime'].dt.dayofweek 
    df['arr_day_of_week'] = df['ArrivalDateTime'].dt.dayofweek 

    df['dep_weekday'] = np.where(df["dep_day_of_week"].isin([5,6]),0,1)
    df['arr_weekday'] = np.where(df["arr_day_of_week"].isin([5,6]),0,1)


    #df['departure_timeOfDay'] = df['dep_hr'].apply(lambda x: time_of_day(x)[0])
    #df['arrival_timeOfDay'] = df['arr_hr'].apply(lambda x: time_of_day(x)[0])
    #df['departure_timeOfDay_encoded'] = df['dep_hr'].apply(lambda x: time_of_day(x)[1])
    #df['arrival_timeOfDay_encoded'] = df['arr_hr'].apply(lambda x: time_of_day(x)[1])

   # one_hot_cols = ['departure_timeOfDay','arrival_timeOfDay']
   # df_oneHotEncoded = pd.get_dummies(df[one_hot_cols])
    #new_df = pd.concat([df,df_oneHotEncoded],axis=1)
    
    new_df = df
    drop_cols = [
                'DepartureDateTime',
                'Duration_timedelta',
                'ArrivalDateTime',
                'Freq_encoded_Source',
                'Mean_encoded_Source',
                'Mean_encoded_Destination',
                #'departure_timeOfDay_encoded',
                'arr_month', 
                'arr_day_of_month',
               # 'departure_timeOfDay_early_morning',
               # 'departure_timeOfDay_evening',
               # 'arrival_timeOfDay_late_night',
               # 'arrival_timeOfDay_morning', 
               # 'arrival_timeOfDay_night'
               ]
    
    #new_df.drop(columns=one_hot_cols,inplace=True)
    new_df.drop(columns=drop_cols,inplace=True)
    return new_df

