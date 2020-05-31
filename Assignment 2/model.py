import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import warnings  
from datetime import datetime
import csv
import copy
import time
import sys
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import xgboost as xgb
from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor,GradientBoostingRegressor
from sklearn import linear_model
from sklearn import svm
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import random
import logging
logging.basicConfig()

#from keras.models import Sequential
#from keras import layers

from sklearn.preprocessing import MinMaxScaler,RobustScaler,StandardScaler
import pdb





b_data = {}

#Pandas options to display rows
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)


#Load the files of
home_c_energy = "data/Home C -2015/homeC2015.csv"
home_c_weather = "data/Home C -2015/HomeC-meter1_2015.csv"

home_b_energy = "data/Home B - 2014/homeB2014.csv"
home_b_weather = "data/Home B - 2014/HomeB-meter1_2014.csv"

home_f_energy = "data/Home F - 2016/homeF2016.csv"
home_f_weather = "data/Home F - 2016/HomeF-meter3_2016.csv"


data_points = {
"home_c" : [home_c_energy,home_c_weather],
"home_b" : [home_b_energy,home_b_weather],
"home_f" : [home_f_energy,home_f_weather]
}

def read_data(home_name) :

    reader = csv.DictReader(open(data_points["home_"+home_name][0],"r"))
    data_extend = []
    for row in reader :                     #Read the weather dataframe
        temp_row = copy.copy(row)
        temp_row['time'] =  int(temp_row['time'])
        data_extend.append(temp_row)
        if home_name in ["b","f","c"] :
            temp_row_2 = copy.copy(row)
            temp_row_2['time'] = int(temp_row_2['time']) + 1800
            data_extend.append(temp_row_2)
    weather_data = pd.DataFrame(data_extend)



    reader = csv.DictReader(open(data_points["home_"+home_name][1],"r"))
    data_extend = []
    for row in reader :                     # Read the meter dataframe, modify timestamps such that it can be merged with the weather data.
        new_row = copy.copy(row)
        if home_name == 'b' :
            new_row['Date & Time'] = datetime.strptime(new_row['Date & Time'],"%m/%d/%Y %H:%M" )
        elif home_name in ['f','c'] :
            new_row['Date & Time'] = datetime.strptime(new_row['Date & Time'],"%Y-%m-%d %H:%M:%S" )
            
        #if new_row['Date & Time'].minute == 30 and home_name in ['b'] :
        #    continue
        #el
        if new_row['Date & Time'].minute not in [0,30] and home_name in ['c'] :
            continue

        new_row['Date & Time'] = int(time.mktime( new_row['Date & Time'].timetuple() ))
        data_extend.append(new_row)
    meter_data = pd.DataFrame(data_extend)



    
    columns = [] 
    for column in list(meter_data.columns.values) : #Capture the meter columns names
        if "kW" in column :
            columns.append(column)

    if home_name in ["f","c"] :
    
        for column in columns :
            meter_data[column] = meter_data[column].apply(lambda x: float(x))           #Convert to float
        meter_data['Date & Time'] = meter_data['Date & Time'].apply(lambda x: datetime.fromtimestamp(x))
        meter_data = meter_data.set_index(['Date & Time'])

        meter_data = meter_data.resample('.5H').sum()                 #Resample the data such that all the minute values are merged into an hour
        meter_data = meter_data.reset_index()
        meter_data['Date & Time'] = meter_data['Date & Time'].apply(lambda x: int(time.mktime( x.timetuple() ))  )


    
    
    data = pd.merge(weather_data,meter_data,how="inner",left_on="time",right_on="Date & Time")      #Merge the weather and mete data
    #data = data.iloc[0:1000,:]
    
    
    
    data = data.drop(columns=['Date & Time'],axis=1)
    data['time'] =  pd.to_datetime(data['time'],unit='s')
    data['completetime'] =  pd.to_datetime(data['time'],unit='s')
    data = data.set_index(['time'])



    for column in columns :
        data[column] = data[column].apply(lambda x: float(x))           #Convert to float

    data['icon'] = pd.factorize(data['icon'], sort=True)[0] + 1             #Convert categorical values to numbers
    data['summary'] = pd.factorize(data['summary'], sort=True)[0] + 1       #Convert categorical values to numbers
    
    cols = ['humidity','temperature','visibility','apparentTemperature','pressure','windSpeed','cloudCover','windBearing','precipIntensity','dewPoint','precipProbability']




    for col in cols :
        data[col] = pd.to_numeric(data[col],errors="raise",downcast='float')


    data = data.fillna(value={'cloudCover':0})



    data['total_energy'] = data.loc[:,columns].sum(axis=1)          #Total energy as a sum of all meter data captured


    #custom aggregator for daily data
    aggregator = {}
    for col in columns :
        aggregator[col] =  np.sum
    for col in cols :
        aggregator[col] = np.mean



     #custom aggregator for daily data
    aggregator['total_energy'] = np.sum
    aggregator['icon'] = np.mean
    aggregator['summary'] = np.mean
    aggregator['is_a_holiday'] = np.mean
    aggregator['season'] = np.mean

    #Check if the day is a holiday and identify the season for the day
    date_range = data['completetime']
    min_date = data.index.min()
    max_date = data.index.max()
    my_calendar = calendar()
    holiday_list = my_calendar.holidays(start=min_date, end=max_date)
    data["is_a_holiday"] = data['completetime'].isin(holiday_list)
    data["is_a_holiday"] = data["is_a_holiday"].map({True: 1, False: 10})
    data['season'] = data.index.dayofyear.map(
        pick_value_for_season
    )
    



    data['dayofweek'] = 0
    count = 0
    daily_dataframe = data.drop(columns=['completetime'])
    for index,row in data.iterrows() :
        data.iloc[count, data.columns.get_loc('dayofweek')] = index.dayofweek
        count += 1



    #Resample data to create data for daily load predictions
    daily_dataframe = daily_dataframe.resample('D').agg(aggregator)
    data = data.drop(columns=columns)
    daily_dataframe  = daily_dataframe.drop(columns=columns)

    daily_dataframe['dayofweek'] = 0
    for index,row in daily_dataframe.iterrows() :
        row['dayofweek'] = index.dayofweek



    #Load next timestep added to each row
    daily_dataframe['load_tomorrow'] = daily_dataframe['total_energy'].shift(-1)
    data['load_next_hour'] = data['total_energy'].shift(-1)



    data = data.dropna()

    daily_data = daily_dataframe



    
    count = 0
    for index,row in daily_dataframe.iterrows() :
        daily_dataframe.iloc[count, daily_dataframe.columns.get_loc('dayofweek')] = index.dayofweek
        count += 1


    data['hour'] = data['completetime'].dt.hour
    data['month'] = data['completetime'].dt.month
    data = data.drop(columns=['completetime'])
    #If energe consumption is less than 10 percentile of the values, assume the family has gone out of the house
    data['Percentile_rank'] = data["total_energy"].rank(pct=True)
    data['family_out'] = data['Percentile_rank'].apply(lambda x: 1 if x < .1 else 10 )
    data = data.drop(columns=['Percentile_rank'])

    daily_data['Percentile_rank'] = daily_data["total_energy"].rank(pct=True)
    daily_data['family_out'] = daily_data['Percentile_rank'].apply(lambda x: 1 if x < .1 else 10 )
    daily_data = daily_data.drop(columns=['Percentile_rank'])



    #data = data.dropna()
    daily_data = daily_data.dropna()




    return [data,daily_data]
    
def train_model(home="b",model_name='linear') :
    temp_data = read_data(home)
    hourly_data = temp_data[0]
    daily_data = temp_data[1]
    
    #Plot the heatmap
    #plot_heatmap(daily_data)
    #plot_heatmap(hourly_data)



    daily_data_x = daily_data.drop(columns=['load_tomorrow'])
    daily_data_y = daily_data['load_tomorrow']
    
    hourly_data_x = hourly_data.drop(columns=['load_next_hour'])
    hourly_data_y = hourly_data['load_next_hour']
    
    #Create the daily data, hourly data train and test suites
    X_train_day, X_test_day, y_train_day, y_test_day = train_test_split(daily_data_x, daily_data_y, test_size=0.2,shuffle=False)
    X_train_hour, X_test_hour, y_train_hour, y_test_hour = train_test_split(hourly_data_x, hourly_data_y, test_size=0.2,shuffle=False)


    #Build naive predictions for the dataset
    daily_naive_preds = naive_model(y_test_day.to_frame(),"daily") #.iloc[:y_test_day.shape[1],:]
    hourly_naive_preds = naive_model(y_test_hour.to_frame(),"hourly") #.iloc[:y_test_hour.shape[1],:]
    
    #Predict the load demand using machine learning models
    #daily_predictions = predict_data(X_train_day,X_test_day,y_train_day,y_test_day,"day",model_name,True,home)
    hourly_predictions = predict_data(X_train_hour,X_test_hour,y_train_hour,y_test_hour,"hour",model_name,True,home )

    #Reshape the incoming data
    #daily_predictions = daily_predictions.reshape(len(daily_predictions),1)
    hourly_predictions = hourly_predictions.reshape(len(hourly_predictions),1)

    #Compute the Mean Absolute Error
    #daily_mae = calculate_mae(y_test_day.to_frame(),daily_predictions,daily_naive_preds,model_name,"hourly")
    #hourly_mae = calculate_mae(y_test_hour.to_frame(),hourly_predictions,hourly_naive_preds,model_name,"hourly")

    return [pd.DataFrame(hourly_predictions, columns=['predictions'], index=y_test_hour.index),None]
    #return [pd.DataFrame(hourly_predictions,columns=['predictions'],index=y_test_hour.index)  ,pd.DataFrame(daily_predictions,columns=['predictions'],index=y_test_day.index)]




    
def predict_data(X_train, X_test, y_train,y_test,data_length="day",model="linear",show_plot=False,home="b") :
    
    print("Predicting energy demand for home "+ home +" using the "+model+" model")
    
    #Linear Regression Model
    if model == "linear" :
        lm = linear_model.LinearRegression()
        model = lm.fit(X_train, y_train)
        predictions = lm.predict(X_test)


    #XGB boose Model
    if model == "xgb" :
        regressor =  xgb.XGBRegressor()
        tscv = TimeSeriesSplit(n_splits=30)
        regressor.fit(X_train,y_train)
        predictions = regressor.predict(X_test)
    
    #RandomForestRegressor Model
    if model == "randomforest" :
        regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
        regressor.fit(X_train,y_train)
        predictions =  regressor.predict(X_test)

    #GradientBoost Model
    if model == "gradientboost" :
        regressor = GradientBoostingRegressor(n_estimators=500,max_depth=4,min_samples_split=2,learning_rate=0.01,loss='ls')
        regressor.fit(X_train,y_train)
        predictions = regressor.predict(X_test)
        
    #Long Short Term Memory model
    if model == "lstm" :
        
        def fetch_model(data,dimension=40) :
            time_series_model = Sequential()
            time_series_model.add(layers.LSTM(dimension,activation="relu",return_sequences=True, input_shape=( data.shape[1] ,1)))
            time_series_model.add(layers.LSTM(dimension,activation="tanh",return_sequences=True))
            time_series_model.add(layers.LSTM(dimension,activation="tanh",return_sequences=False))
            time_series_model.add(layers.Dropout(3))
            time_series_model.add(layers.Dense(1))
            
            return time_series_model
        #time_series_model.summary()
         
        sequence_length = X_train.shape[1]
        
        data_length_value = X_test.index.values
        load_value = y_test.values
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        X_train = np.reshape(X_train, (X_train.shape[0], sequence_length, 1))
        X_test = np.reshape(X_test, (X_test.shape[0], sequence_length, 1))
        X_train = np.reshape(X_train, (X_train.shape[0], sequence_length, 1))

        
        if data_length != 'day' :
            num_epochs = 150
        else :
            num_epochs = 500
        
        time_series_model = fetch_model(X_train)
        time_series_model.compile(optimizer="adam",loss="MAE")
        time_series_model.fit(X_train, y_train, epochs=num_epochs, batch_size=20)

        
        predictions = time_series_model.predict(X_test)
        temp_preds = []
        for val in predictions :
            temp_preds.append(val[0])
        predictions = np.array(temp_preds)
    
    #Plot actual load vs predicted load on to a graph
    '''
    if model == "lstm" :
        plot_data = pd.DataFrame({ data_length : data_length_value ,"Actual load": load_value ,"Predicted Load":predictions })
    else :
        plot_data = pd.DataFrame({ data_length : X_test.index.values,"Actual load": y_test.values,"Predicted Load":predictions.tolist() })
    if show_plot :
        plot_actualvspredicted(plot_data,data_length,home)
    '''
    return predictions


def calculate_mae(y,preds,naive_preds,model,type="daily") :
   '''
   Compute the mean absolute error
   
   '''
   
   
   count = 0
   total_length = y.shape[0]
   naive_mae_sum = 0
   preds_mae_sum = 0
   
   if type == "daily" :
    field = "load_tomorrow"
   else :
    field = "load_next_hour"
   

   for index,row in y.iterrows() :
    try :
        y_test = y.iloc[[count]].values.tolist()[0][0]
        ny = naive_preds.iloc[[count]].values.tolist()[0][0]
        py = preds[count][0]

        naive_mae_sum = naive_mae_sum + abs( y_test - ny  )
        preds_mae_sum = preds_mae_sum + abs( y_test - py )
        count += 1
    except Exception as e :
        print(e)
        sys.exit(1)
    
   print("Mean absolute error for the naive model is {}".format( str( naive_mae_sum/total_length  ) ))
   print("Mean absolute error for the {} model is {}".format( model,str( preds_mae_sum/total_length  ) ))
    

def plot_heatmap(data,xsize=10,ysize=5):
    '''
    Plot the correlation heatmap for the datapoints attributes
    '''
    plt.figure(figsize=(xsize,ysize))
    sns.heatmap(data.corr())
    plt.show()
    
def plot_actualvspredicted(dataframe,type="day",home=""):
    '''
    Plot for actual vs predicted values
    '''
    dataframe.set_index(type).plot(figsize=(10,5), grid=True)
    plt.xlabel("TIME")
    plt.ylabel("Load in KW")
    plt.title("Actual vs Predicted load for home "+home)
    plt.show()
    

def pick_value_for_season(day) :
    '''
    Identify the season for the current day
    '''
    autumm = range(1,80)
    spring = range(80, 172)
    summer = range(172, 264)
    fall = range(264, 355)
    
    if day in spring or day in autumm:
       return 0
    else :
        return 10

def naive_model(dataframe,type="daily") :
    '''
    The naive model simply picks up a random total energy demand from a row 
    and sets it as the predicted load either for an hour or day.
    '''
    
 
    
    count = 1
    naive_preds = []
    
    shape = dataframe.shape[0]
    try :
        if type == "daily" :
            for index,row in dataframe.iterrows() :
                
                rand = random.randint(1,shape-1)
                
                naive_preds.append({ "day" : index,  "load_tomorrow" : dataframe.iloc[[rand]]['load_tomorrow'].values[0]   }) #
                count += 1 
            naive_preds = pd.DataFrame(naive_preds)
            naive_preds = naive_preds.set_index("day")
        else :
            for index,row in dataframe.iterrows() :
                rand = random.randint(1,shape-1)
                naive_preds.append({ "hour" : index,  "load_next_hour" :  dataframe.iloc[[rand]]['load_next_hour'].values[0]  }) #
                count += 1
            naive_preds = pd.DataFrame(naive_preds)
            naive_preds = naive_preds.set_index("hour")
    except Exception as e :
        print(e)

    
    return naive_preds




