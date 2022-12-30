
import sys
sys.path.append('src')#no need to import src funcs if we use this
# https://stackoverflow.com/questions/4761041/python-import-src-modules-when-running-tests


from flask import Flask, json, request, jsonify , render_template
import os
import urllib.request
from werkzeug.utils import secure_filename
#

import pandas as pd
import numpy as np 

from predict import encode_predict_input, predict_price


#from predict import preprocess_and_predict

from variables import airlines, sources, destinations, routes

import joblib
import pickle

app = Flask(__name__)

#model= joblib.load(model_path)

@app.route('/')
def home():
    return "Welcome to Home Page"


@app.route('/api', methods=['POST','GET'])
def predict():
    # Get the data from the POST request.
    json_data = request.get_json(force=True)

    # Convert json data to dataframe
    df = pd.DataFrame.from_dict(pd.json_normalize(json_data), orient='columns')
    print("-"*80)
    print(df)

    # Pre-process and make prediction using model loaded from disk as per the data.
    #data = preprocess_and_predict(df,encoded_dict)
    #print("-"*80)
    #print(data)
    #print("-"*80)
    
    model_path = 'models/xgboost_demo.pickle'
    encoded_path = 'models/encoded_dict_demo.pickle'
    prediction = predict_price(df, encoded_path, model_path)


    # Take the first value of prediction
    output = prediction[0]
    print("price : ",output)
    return jsonify({"price" : str(output)})

# @app.route("/")
# def root():
#     return 'Root of Flask WebApp!'


#@app.route("/prediction")
#def prediction():

 #  Source = request.args.get('Source')
  #  Destination = request.args.get('Destination')
   # Departure_Date = request.args.get('Departure_Date')
    #Arrival_Date = request.args.get('Arrival_Date')
    #Departure_Time = request.args.get('Departure_Time')
    #Arrival_Time = request.args.get('Arrival_Time')
   # Route = request.args.get('Route')
    #Stops = request.args.get('Stops')
    #Additional = request.args.get('Additional')

    #data_dict = {
     #       'Airline'  : Airline,
      #      'Source'  : Source,
       #     'Destination'  : Destination,
        #    'Date_of_Journey'  : Departure_Date,
         #   'Dep_Time'  : Departure_Time,
            # 'Arrival_Date'  : Arrival_Date,
          #  'Duration'  : '2h 50m',
           # 'Arrival_Time'  : Arrival_Time,
            #'Route'  : Route,
           # 'Total_Stops' : Stops,
           # 'Additional_Info'  : Additional,
        #}

    #print(data_dict)

    # Convert json data to dataframe
    #df = pd.DataFrame.from_dict([data_dict],orient='columns')
    # print("-"*80)
   # print(df)

    # Pre-process and make prediction using model loaded from disk as per the data.
    #data = preprocess_and_predict(df,encoded_dict)
    #print("-"*80)
    #print(data)
    #print("-"*80)
 
    #prediction = model.predict(data)


    #return str(prediction[0])

@app.route('/index', methods=['GET', 'POST'])
def index():
    return render_template('index.html')
  # airlines=airlines, 
  #  sources=sources, destinations=destinations,
   # routes=routes , pred=str(9999))

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)