import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import sparse
import datetime
import geopy.distance

######codes for loading historical original airquality and weather data
def load_data_with_weather(start_17='2017-02-14 00:00:00',end_17='2017-07-01 00:00:00',start_18='2018-02-01 00:00:00',end_18='2018-04-01 00:00:00'):
    '''
        This function is used to load air pollution index together with info in a uniform format.
    '''
    #load the historial data in 2017 and 2018
    #here I use data from 2017-02-14-2017-07-01 and 2018-02-01 2018-04-01
    weather_used_old = ['pressure','temperature','humidity','wind_direction','wind_speed/kph']
    aq_old,ob_old,grid_old = load_old_all(start_17,end_17,start_18,end_18)

    #prepare weather data
    weather_old_all = pd.concat([ob_old,grid_old]).reset_index()
    weather_old_all = clean_weather(weather_old_all)
    weather_old_all = weather_old_all.drop(['latitude','longitude'],axis=1)
    weather_old_all = weather_old_all.drop_duplicates(subset=['utc_time','station_id'])

    #add time columns to airQuality data
    #levels is the number of nearest weather station to search
    #time_column is the name of the time column
    print("before fill: ",aq_old.shape[0])
    aq_old_all = fill_weather_gap(aq_df=aq_old,weather_df=weather_old_all,levels = 1,time_column = 'utc_time',weather_attrs=weather_used_old)
    print("after fill: ",aq_old_all.shape[0])

    aq_old_all['timestamp'] = pd.to_datetime(aq_old_all['utc_time'])
    aq_old_all['year'] = aq_old_all['timestamp'].dt.year
    aq_old_all['month'] = aq_old_all['timestamp'].dt.month
    aq_old_all['day'] = aq_old_all['timestamp'].dt.day
    aq_old_all['hour'] = aq_old_all['timestamp'].dt.hour
    
    weather_used = ['pressure','temperature','humidity','wind_direction','wind_speed']
    aq_df4 = pd.read_csv('../data/aiqQuality_201804.csv')
    grid_df4 = pd.read_csv('../data/gridWeather_201804.csv')
    ob_df4 = pd.read_csv('../data/observedWeather_201804.csv')

    test_grid = pd.read_csv('../data/gridWeather_20180501-20180502.csv')
    test_weather = pd.read_csv('../data/observedWeather_20180501-20180502.csv')
    airq_5= generate(aq_df4,'2018-05-01 00:00:00','2018-05-02 23:00:00')
    #print("aq5 before:",airq_5.shape)
    gt = pd.read_csv('../data/external_0501-0502.csv')
    gt = gt.sort_values(by='time').reset_index(drop=True)
    airq_5 = airq_5[['time','station_id']].merge(gt,how='left',on=['station_id','time'])
    #print("aq5 after:",airq_5.shape)
    aq_df4 = aq_df4.append(airq_5)



    #prepare weather data
    weather_all = pd.concat([ob_df4,grid_df4,test_grid,test_weather]).reset_index()
    weather_all = clean_weather(weather_all)
    
    #add time columns to airQuality data
    #levels is the number of nearest weather station to search
    #time_column is the name of the time column
    aq_all = fill_weather_gap(aq_df=aq_df4,weather_df=weather_all,levels = 1,time_column = 'time',weather_attrs=weather_used)


    aq_all['timestamp'] = pd.to_datetime(aq_all['time'])
    aq_all['year'] = aq_all['timestamp'].dt.year
    aq_all['month'] = aq_all['timestamp'].dt.month
    aq_all['day'] = aq_all['timestamp'].dt.day
    aq_all['hour'] = aq_all['timestamp'].dt.hour
    
    #adjust different columns names
    old_columns = aq_all.columns
    new_columns = [x[:-14] if x.endswith('_Concentration') else x for x in old_columns]
    aq_all.columns = new_columns

    #add geo info of the air quality stations for nearest weather station searching
    aq_geo = pd.read_csv('../data/aq_geo.csv')
    aq_all = aq_all.merge(aq_geo,how='left',on='station_id')
    aq_all = aq_all.drop(['id'],axis=1)
    
    #update the name of historical data
    old_columns = list(aq_old_all.columns)
    old_columns[old_columns.index('PM2.5')] = 'PM25'
    old_columns[old_columns.index('wind_speed/kph')] = 'wind_speed'
    old_columns[old_columns.index('utc_time')] = 'time'
    aq_old_all.columns = old_columns
    
    #seperate data into 2017 and 2018
    aq_2017 = aq_old_all.loc[aq_old_all.year == 2017]
    aq_2018_old = aq_old_all.loc[aq_old_all.year == 2018]
    aq_2018 = pd.concat([aq_2018_old,aq_all],axis=0)
    
    weather_old_all['time'] = weather_old_all['utc_time']
    weather_all_20172018 = pd.concat([weather_old_all[['station_id','time','weather']],weather_all[['station_id','time','weather']]],axis=0)
    old_columns = list(weather_all_20172018.columns)
    old_columns[old_columns.index('station_id')] = 'w_end'
    weather_all_20172018.columns = old_columns
    
    aq_2018 = aq_2018.merge(weather_all_20172018,how='left',on=['w_end','time'])
    aq_2017 = aq_2017.merge(weather_all_20172018,how='left',on=['w_end','time'])

    #prepare bad weather dict
    weather_keys = list(set(aq_2018.weather.unique())|set(aq_2017.weather.unique()))
    bad_weathers = ['Sleet','HAZE','Hail','Haze']
    w_dict = {}
    for w in weather_keys:
        if w in bad_weathers:
            w_dict[w] = 1
        else:
            w_dict[w] = 0
            
    aq_2018 = aq_2018.set_index(pd.DatetimeIndex(aq_2018['timestamp']))
    aq_2017 = aq_2017.set_index(pd.DatetimeIndex(aq_2017['timestamp']))
    aq_2018['weekday'] = aq_2018['timestamp'].dt.weekday
    aq_2017['weekday'] = aq_2017['timestamp'].dt.weekday
    aq_2017 = aq_2017.drop_duplicates(subset = ['station_id','time'])
    aq_2018 = aq_2018.drop_duplicates(subset=['station_id','time'])
    

    #add the label for bad weather
    aq_2018['bad_weather'] = aq_2018['weather'].apply(lambda x:w_dict[x])
    aq_2017['bad_weather'] = aq_2017['weather'].apply(lambda x:w_dict[x])
    
    aq_2018['is_rain'] = aq_2018['weather'].apply(lambda x: 1 if x == 'RAIN' else 0)
    aq_2017['is_rain'] = aq_2017['weather'].apply(lambda x: 1 if x == 'Rain' else 0)
    
    return aq_2018,aq_2017


def get_df_in_range(df,start,end,attr_name):
    return df.loc[(df[attr_name] >= start) & (df[attr_name] < end)]

def load_old_all(start_17='2017-02-14 00:00:00',end_17='2017-07-01 00:00:00',start_18='2018-02-01 00:00:00',end_18='2018-04-01 00:00:00'):
    '''
        Load all data needed in 2017.02.14-2017.06.30 & 2018.02.01 - 2018.03.31
    '''
    aq_year_df = pd.read_csv('../data/airQuality_201701-201801.csv')
    grid_year_df = pd.read_csv('../data/gridWeather_201701-201803.csv')
    ob_year_df = pd.read_csv('../data/observedWeather_201701-201801.csv')

    aq_year_needed = get_df_in_range(aq_year_df,start_17, end_17,'utc_time')
    grid_year_needed = get_df_in_range(grid_year_df,start_17, end_17,'utc_time')
    ob_year_needed = get_df_in_range(ob_year_df,start_17, end_17,'utc_time')

    aq_year_df1 = pd.read_csv('../data/airQuality_201802-201803.csv')
    ob_year_df1 = pd.read_csv('../data/observedWeather_201802-201803.csv')
    grid_year_df1 = get_df_in_range(grid_year_df,start_18, end_18,'utc_time')
    aq_year_df1 = get_df_in_range(aq_year_df1,start_18, end_18,'utc_time')
    ob_year_df1 = get_df_in_range(ob_year_df1,start_18,end_18,'utc_time')

    aq_old = pd.concat([aq_year_needed,aq_year_df1],axis=0)
    ob_old = pd.concat([ob_year_needed,ob_year_df1],axis=0)
    grid_old = pd.concat([grid_year_needed,grid_year_df1],axis=0)

    ob_old = ob_old.drop(['latitude','longitude'],axis=1)
    grid_old['station_id'] = grid_old['stationName']
    grid_old = grid_old.drop(['stationName'],axis=1)
    
    #load geo df
    aq_geo = pd.read_csv('../data/aq_geo.csv')
    ob_geo = pd.read_csv('../data/observeWeather_geo.csv')
    
    aq_old = aq_old.merge(aq_geo,how='left',left_on=['stationId'],right_on=['station_id'])
    aq_old  = aq_old.drop(['stationId'],axis=1)
    
    ob_old = ob_old.merge(ob_geo[['latitude','longitude','station_id']],how='left',on='station_id')

    return aq_old,ob_old,grid_old
	
######code for filling the weather info
def get_aq_w_dist():
    aq_geo = pd.read_csv('../data/aq_geo.csv')[['station_id','latitude','longitude']].values
    obs_geo = pd.read_csv('../data/observeWeather_geo.csv')[['station_id','latitude','longitude']].values
    grid_geo = pd.read_csv('../data/Beijing_grid_weather_station.csv',names=['station_id','latitude','longitude'])[['station_id','latitude','longitude']].values
    
    aq_start = []
    w_end = []
    dist_list = []
    for i in range(aq_geo.shape[0]):
        for j in range(obs_geo.shape[0]):
            aq_start.append(aq_geo[i][0])
            w_end.append(obs_geo[j][0])
            dist_list.append(geopy.distance.vincenty((aq_geo[i][1],aq_geo[i][2]),(obs_geo[j][1],obs_geo[j][2])).km)

        for k in range(grid_geo.shape[0]):
            aq_start.append(aq_geo[i][0])
            w_end.append(grid_geo[k][0])
            dist_list.append(geopy.distance.vincenty((aq_geo[i][1],aq_geo[i][2]),(grid_geo[k][1],grid_geo[k][2])).km)
    aq_w_dist_df = pd.DataFrame({'aq_start':aq_start,'w_end':w_end,'dist':dist_list})
    aq_w_dist_df = aq_w_dist_df.sort_values(by=['aq_start','dist'])
    
    #get top 3 nearest station for each aq_station
    top3_dict = {}
    aq_w_grouped = aq_w_dist_df.groupby('aq_start')
    for key,group in aq_w_grouped:
        top3_dict[key] = group['w_end'].values[:3]
    aq_group = aq_w_dist_df.groupby('aq_start',as_index=False)
    return aq_group

def clean_weather(df):
    df = df.loc[df.humidity < 100]
    df = df.loc[df.pressure < 3000]
    df = df.loc[df.temperature < 60]
    return df
	
def add_nearest_weather(aq_df,weather_all,aq_w_dist_group,level,time_column,weather_attrs = [ 'humidity', 'pressure', 'temperature']):
    '''
        This function find the nearesr weather station for the air quality data
    '''
    if level == 0:
        print("level 0 before fill: ",aq_df.shape[0]) 	
        new_df1 = pd.merge(aq_df, aq_w_dist_group.nth(0),  how='left', left_on=['station_id'], right_on = ['aq_start'])
        aq_all = pd.merge(new_df1, weather_all[weather_attrs+['station_id',time_column]],  how='left', left_on=['w_end',time_column], right_on = ['station_id',time_column])
        print("level 1 after fill: ",aq_all.shape[0])
        return aq_all
    else:
        #print("level"+str(level)+" before fill: ",aq_df.shape[0])
        aq_na_df = aq_df[aq_df[weather_attrs].isnull().any(axis=1)]
        aq_not_na_df = aq_df[~aq_df[weather_attrs].isnull().any(axis=1)]
        print("not na vs na:",aq_not_na_df.shape[0],aq_na_df.shape[0])    
        to_drop = ['aq_start','dist', 'w_end', 'id_y', 'station_id_y']+weather_attrs
        real_drop = [x for x in to_drop if x in aq_na_df.columns]    
        aq_na_df = aq_na_df.drop(real_drop,axis=1)
        old_columns = aq_na_df.columns
        new_columns = [x[:-2] if x.endswith('_x') else x for x in old_columns]

        aq_na_df.columns = new_columns 
        
        new_df2 = pd.merge(aq_na_df,aq_w_dist_group.nth(level),  how='left', left_on=['station_id'], right_on = ['aq_start'])
        aq_na_df_new = pd.merge(new_df2, weather_all[weather_attrs+['station_id',time_column]],  how='left', left_on=['w_end',time_column], right_on = ['station_id',time_column])
        aq_new_all = pd.concat([aq_not_na_df,aq_na_df_new],axis=0)
        #print("level"+str(level)+" after fill: ",aq_new_all.shape[0])
        #print(aq_not_na_df.columns)
        #print(aq_na_df_new.columns)
        return aq_new_all
		
def fill_weather_gap(aq_df,weather_df,levels,time_column,weather_attrs=['humidity','pressure','temperature']):
    aq_group = get_aq_w_dist()
    aq_all = add_nearest_weather(aq_df,weather_df,aq_group,0,time_column,weather_attrs)
    for i in range(1,levels):
        print("Level "+str(i)+" #null before",aq_all[weather_attrs].isnull().any(axis=1).sum())
        aq_all = add_nearest_weather(aq_all,weather_df,aq_group,i,time_column,weather_attrs)
        print("After: ",aq_all[weather_attrs].isnull().any(axis=1).sum())
    aq_all['station_id'] = aq_all['station_id_x']
    aq_all = aq_all.drop(['station_id_x','station_id_y','aq_start'],axis=1)
    return aq_all
	
def generate(dataframe,start='2018-05-01 00:00:00',end='2018-05-02 22:00:00'):
    stations = dataframe['station_id'].unique()
    columns = list(dataframe.columns)
    whole=None
    for i in list(stations):
        #print(i)
        section = pd.DataFrame({'time':pd.date_range(start=start, end=end, freq='1h')})
        #print(section)
        section['time']=section['time'].apply(lambda x: str(x))
        for k in columns:
            if k == 'station_id':
                section[k] = i
            elif k == 'time':
                continue
            else:
                section[k]=0               
        try:
            whole = whole.append(section)
        except:
            whole = section
        #print(whole)
    return whole.sort_values(by=['time'])