list_A = ['Total_Stops', 'Duration_min', 'Freq_encoded_Airline',
       'Freq_encoded_Destination', 'Freq_encoded_Route',
       'Mean_encoded_Airline', 'Mean_encoded_Route', 'Label_encoded_Airline',
       'Label_encoded_Source', 'Label_encoded_Destination',
       'Label_encoded_Route', 'dep_hr', 'arr_hr', 'dep_month',
       'dep_day_of_month', 'dep_day_of_week', 'arr_day_of_week', 'dep_weekday',
       'arr_weekday', 'arrival_timeOfDay_encoded'
    ] 

list_B = ['Total_Stops', 'Duration_min', 'dep_hr', 'arr_hr', 'dep_month',
       'dep_day_of_month', 'dep_day_of_week', 'arr_day_of_week', 'dep_weekday',
       'arr_weekday', 'arrival_timeOfDay_encoded', 'Freq_encoded_Airline',
       'Mean_encoded_Airline', 'Label_encoded_Airline', 'Label_encoded_Source',
       'Freq_encoded_Destination', 'Label_encoded_Destination',
       'Freq_encoded_Route', 'Mean_encoded_Route', 'Label_encoded_Route'
      ]

print(len(list_A))
print(len(list_B))

print(set(list_A) - set(list_B))