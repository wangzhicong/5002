import lightgbm
import numpy as np
import pandas as pd
from features_extraction import *

def smape(actual, predicted):
    dividend= np.abs(np.array(actual) - np.array(predicted))
    denominator = np.array(actual) + np.array(predicted)
    
    return 2 * np.mean(np.divide(dividend, denominator, out=np.zeros_like(dividend), where=denominator!=0, casting='unsafe'))

def get_features_model_dict(aq_2017,aq_2018):
    '''
    This function is used to prepare models and data for predict pollution index in the next 48 hours 
    '''
    aq_2017['date_h'] = aq_2017['timestamp'].apply(lambda x: (x.timetuple().tm_yday)*24 + x.hour + 24)
    aq_2018['date_h'] = aq_2018['timestamp'].apply(lambda x: (x.timetuple().tm_yday)*24+(x.hour) )

    #step1: Prepare data for model training
    aq_2017_O3 = aq_2017.loc[aq_2017.time > '2018-03-28 00:00:00']
    aq_train_O3 = aq_2018.loc[(aq_2018.time < '2018-05-01 00:00:00')]
    aq_test_O3 = aq_2018.loc[aq_2018.time >= '2018-05-01 00:00:00']
    aq_train_O3 = pd.concat([aq_2017_O3,aq_train_O3],axis=0)
    fill_aq_O3,fill_date_O3 = fill_time_gap(aq_train_O3)
    filled_features = ['O3']
    fill_aq_O3 = fill_null_attrs(fill_aq_O3,filled_features)
    fill_aq_O3 = fill_weather_by_mode(fill_aq_O3)
    #prepare one hot info for stations
    fill_aq_O3,stations_col = one_hot_stations(fill_aq_O3)
    fill_df_O3,attrs_all_O3 = add_features_all(fill_aq_O3,'O3')

    aq_train_PM10 = aq_2018.loc[(aq_2018.time < '2018-05-01 00:00:00')&(aq_2018.time > '2018-03-28 00:00:00')]
    aq_test_PM10 = aq_2018.loc[aq_2018.time >= '2018-05-01 00:00:00']
    fill_aq_PM10,fill_date_PM10 = fill_time_gap(aq_train_PM10)
    filled_features = ['PM10']
    fill_aq_PM10 = fill_null_attrs(fill_aq_PM10,filled_features)
    fill_aq_PM10 = fill_weather_by_mode(fill_aq_PM10)
    #remove PM10 outlier(this days all have extremely high value):
    fill_aq_PM10 = fill_aq_PM10.loc[(fill_aq_PM10.time < '2018-03-27 22:00:00')|(fill_aq_PM10.time > '2018-03-28 14:00:00')]
    #prepare one hot info for stations
    fill_aq_PM10,stations_col = one_hot_stations(fill_aq_PM10)
    fill_df_PM10,attrs_all_pm10 = add_features_all(fill_aq_PM10,'PM10')

    aq_train_PM25 = aq_2018.loc[(aq_2018.time < '2018-05-01 00:00:00') & (aq_2018.time > '2018-04-10 00:00:00') ]
    aq_test_PM25 = aq_2018.loc[aq_2018.time >= '2018-05-01 00:00:00']
    fill_aq_PM25,fill_date = fill_time_gap(aq_train_PM25)
    filled_features = ['PM25']
    target= ['PM25']
    mean = fill_aq_PM25[target].mean().values[0]
    print(mean)
    #fill_aq[target] = fill_aq[target].apply(lambda x: x - mean )
    fill_aq_PM25 = fill_null_attrs(fill_aq_PM25,filled_features)
    fill_aq_PM25 = get_rolling_fill(fill_aq_PM25,filled_features)
    fill_aq_PM25 = fill_weather_by_mode(fill_aq_PM25)
    #prepare one hot info for stations
    fill_aq_PM25,stations_col = one_hot_stations(fill_aq_PM25)
    fill_df_PM25,attrs_all_pm25 = add_features_all(fill_aq_PM25,'PM25')


    fill_df_O3 = fill_df_O3.loc[fill_df_O3.station_id != 'zhiwuyuan_aq']
    fill_df_PM25 = fill_df_PM25.loc[fill_df_PM25.station_id != 'zhiwuyuan_aq' ]
    fill_df_PM10 = fill_df_PM10.loc[fill_df_PM10.station_id != 'zhiwuyuan_aq']

    train_df_O3 = fill_df_O3.loc[fill_df_O3.time < '2018-04-29 00:00:00']
    val_df_O3 = fill_df_O3.loc[fill_df_O3.time >= '2018-04-29 00:00:00']
    train_df_PM10 = fill_df_PM10.loc[fill_df_PM10.time < '2018-04-29 00:00:00']
    val_df_PM10 = fill_df_PM10.loc[fill_df_PM10.time >= '2018-04-29 00:00:00']
    train_df_PM25 = fill_df_PM25.loc[fill_df_PM25.time < '2018-04-29 00:00:00']
    val_df_PM25 = fill_df_PM25.loc[fill_df_PM25.time >= '2018-04-29 00:00:00']

    train_df_O3[attrs_all_O3] = train_df_O3[attrs_all_O3].fillna(method='ffill')
    train_df_PM10[attrs_all_pm10] = train_df_PM10[attrs_all_pm10].fillna(method='ffill')
    train_df_PM25[attrs_all_pm25] = train_df_PM25[attrs_all_pm25].fillna(method='ffill')

    date_cols_O3 = ['year','month','day','hour','weekday','date_h']
    weather_used_O3 = ['pressure','temperature','humidity','wind_speed','bad_weather']
    train_x_O3 = train_df_O3[attrs_all_O3+list(stations_col)+date_cols_O3+weather_used_O3].values
    train_y_O3 = train_df_O3['O3_filled'].values
    val_x_O3 = val_df_O3[attrs_all_O3+list(stations_col)+date_cols_O3+weather_used_O3].values
    val_y_O3 = val_df_O3['O3_filled'].values

    date_cols_PM10 = ['year','month','day','hour','weekday','date_h']
    weather_used_PM10 = ['pressure','temperature','humidity','wind_speed']
    train_x_PM10 = train_df_PM10[attrs_all_pm10+list(stations_col)+date_cols_PM10+weather_used_PM10].values
    train_y_PM10 = train_df_PM10['PM10_filled'].values
    val_x_PM10 = val_df_PM10[attrs_all_pm10+list(stations_col)+date_cols_PM10+weather_used_PM10].values
    val_y_PM10 = val_df_PM10['PM10_filled'].values


    date_cols_PM25 = ['year','month','day','hour','weekday']
    weather_used_PM25 = ['pressure','temperature','humidity','wind_speed']
    train_x_PM25 = train_df_PM25[attrs_all_pm25+list(stations_col)+date_cols_PM25+weather_used_PM25].values
    train_y_PM25 = train_df_PM25['PM25_filled'].values
    val_x_PM25 = val_df_PM25[attrs_all_pm25+list(stations_col)+date_cols_PM25+weather_used_PM25].values
    val_y_PM25 = val_df_PM25['PM25_filled'].values
    
    #Step2 train the models
    train_O3 = lightgbm.Dataset(train_x_O3,train_y_O3)
    train_PM10 = lightgbm.Dataset(train_x_PM10,train_y_PM10)
    train_PM25 = lightgbm.Dataset(train_x_PM25,train_y_PM25)


    #train LightGBM models for each pollution index
    PM25_param = {  
            'boosting_type': 'gbdt',  
            'objective': 'mse',  
            'metric': {'mse'},    
            'max_depth': 20,  
            'learning_rate': 0.01,  
            'feature_fraction': 0.9,  
            'bagging_fraction': 0.95,  
            'lambda_l2': 0.001,    
            'verbose': 0,
        }
    PM10_param = {  
            'boosting_type': 'gbdt',  
            'objective': 'mse',  
            'metric': {'mse'},    
            'max_depth': 10,  
            'learning_rate': 0.01,  
            'feature_fraction': 0.9,  
            'bagging_fraction': 0.95,  
            'lambda_l2': 0.001,    
            'verbose': 4,
        }  

    O3_param =  {  
            'boosting_type': 'gbdt',  
            'objective': 'mse',  
            'metric': {'mse'},  
            'max_depth': 8,  
            'learning_rate': 0.01,  
            'feature_fraction': 0.9,  
            'bagging_fraction': 0.95,   
            'lambda_l2': 0.001,  
            'verbose': 4,
        } 

    O3_model = lightgbm.train(O3_param,train_O3,num_boost_round = 400,valid_sets=[train_O3],valid_names = ['train'])
    PM10_model = lightgbm.train(PM10_param,train_PM10,num_boost_round = 400,valid_sets=[train_PM10],valid_names = ['train'])
    PM25_model = lightgbm.train(PM25_param,train_PM25,num_boost_round = 400,valid_sets=[train_PM25],valid_names = ['train'])
    
    O3_pred = O3_model.predict(val_x_O3)
    PM10_pred = PM10_model.predict(val_x_PM10)
    PM25_pred = PM25_model.predict(val_x_PM25)
    print("O3 validation smape",smape(val_y_O3,O3_pred))
    print("PM10 validation smape",smape(val_y_PM10,PM10_pred))
    print("PM25 validation smape",smape(val_y_PM25,PM25_pred))
    
    aq_test_PM25 = aq_test_PM25.loc[aq_test_PM25.station_id != 'zhiwuyuan_aq']
    aq_test_PM25,stations_col = one_hot_stations(aq_test_PM25)
    aq_test_PM10 = aq_test_PM10.loc[aq_test_PM10.station_id != 'zhiwuyuan_aq']
    aq_test_PM10,stations_col = one_hot_stations(aq_test_PM10)
    aq_test_O3 = aq_test_O3.loc[aq_test_O3.station_id != 'zhiwuyuan_aq']
    aq_test_O3,stations_col = one_hot_stations(aq_test_O3)
    
    attrs_all_PM25 = attrs_all_pm25+list(stations_col)+date_cols_PM25+weather_used_PM25
    attrs_all_PM10 = attrs_all_pm10+list(stations_col)+date_cols_PM10+weather_used_PM10
    attrs_all_O3 = attrs_all_O3+list(stations_col)+date_cols_O3+weather_used_O3
    
    O3_dict = {'models':O3_model,'aq_test':aq_test_O3,'train_df':train_df_O3,'val_df':val_df_O3,'attrs_used':attrs_all_O3,'attr_pred':'O3'}
    PM10_dict = {'models':PM10_model,'aq_test':aq_test_PM10,'train_df':train_df_PM10,'val_df':val_df_PM10,'attrs_used':attrs_all_PM10,'attr_pred':'PM10'}
    PM25_dict = {'models':PM25_model,'aq_test':aq_test_PM25,'train_df':train_df_PM25,'val_df':val_df_PM25,'attrs_used':attrs_all_PM25,'attr_pred':'PM25'}
    
    return O3_dict,PM10_dict,PM25_dict