import pandas as pd
import numpy as np
import geopy.distance
from datetime import timedelta

######codes for loading historical original airquality and weather data
def get_df_in_range(df,start,end,attr_name):
    return df.loc[(df[attr_name] >= start) & (df[attr_name] < end)]

def load_old_all(start_17='2017-02-14 00:00:00',end_17='2017-07-01 00:00:00',start_18='2018-02-01 00:00:00',end_18='2018-04-01 00:00:00'):
    '''
        Load all data needed in 2017.02.14-2017.06.30 & 2018.02.01 - 2018.03.31
    '''
    aq_year_df = pd.read_csv('./data/airQuality_201701-201801.csv')
    grid_year_df = pd.read_csv('./data/gridWeather_201701-201803.csv')
    ob_year_df = pd.read_csv('./data/observedWeather_201701-201801.csv')

    aq_year_needed = get_df_in_range(aq_year_df,start_17, end_17,'utc_time')
    grid_year_needed = get_df_in_range(grid_year_df,start_17, end_17,'utc_time')
    ob_year_needed = get_df_in_range(ob_year_df,start_17, end_17,'utc_time')

    aq_year_df1 = pd.read_csv('./data/airQuality_201802-201803.csv')
    ob_year_df1 = pd.read_csv('./data/observedWeather_201802-201803.csv')
    grid_year_df1 = get_df_in_range(grid_year_df,start_18, end_18,'utc_time')

    aq_old = pd.concat([aq_year_needed,aq_year_df1],axis=0)
    ob_old = pd.concat([ob_year_needed,ob_year_df1],axis=0)
    grid_old = pd.concat([grid_year_needed,grid_year_df1],axis=0)

    ob_old = ob_old.drop(['latitude','longitude'],axis=1)
    grid_old['station_id'] = grid_old['stationName']
    grid_old = grid_old.drop(['stationName'],axis=1)
    
    #load geo df
    aq_geo = pd.read_csv('./data/aq_geo.csv')
    ob_geo = pd.read_csv('./data/observeWeather_geo.csv')
    
    aq_old = aq_old.merge(aq_geo,how='left',left_on=['stationId'],right_on=['station_id'])
    aq_old  = aq_old.drop(['stationId'],axis=1)
    
    ob_old = ob_old.merge(ob_geo[['latitude','longitude','station_id']],how='left',on='station_id')

    return aq_old,ob_old,grid_old
	
######code for filling the weather info
def get_aq_w_dist():
    aq_geo = pd.read_csv('./data/aq_geo.csv')[['station_id','latitude','longitude']].values
    obs_geo = pd.read_csv('./data/observeWeather_geo.csv')[['station_id','latitude','longitude']].values
    grid_geo = pd.read_csv('./data/Beijing_grid_weather_station.csv',names=['station_id','latitude','longitude'])[['station_id','latitude','longitude']].values
    
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
	
def add_nearest_weather(aq_df,weather_all,aq_w_dist_group,level,time_column):
    '''
        This function find the nearesr weather station for the air quality data
    '''
    weather_attrs = [ 'humidity', 'pressure', 'temperature']
    if level == 0:
        new_df1 = pd.merge(aq_df, aq_w_dist_group.nth(0),  how='left', left_on=['station_id'], right_on = ['aq_start'])
        aq_all = pd.merge(new_df1, weather_all[weather_attrs+['station_id',time_column]],  how='left', left_on=['w_end',time_column], right_on = ['station_id',time_column])
        return aq_all
    else:
        aq_na_df = aq_df[aq_df[weather_attrs].isnull().any(axis=1)]
        aq_not_na_df = aq_df[aq_df[weather_attrs].notnull().any(axis=1)]
        to_drop = ['aq_start','dist', 'w_end','humidity', 'id_y', 'pressure', 'station_id_y', 'temperature','weather', 'wind_direction', 'wind_speed']
        real_drop = [x for x in to_drop if x in aq_na_df.columns]    
        aq_na_df = aq_na_df.drop(real_drop,axis=1)
        old_columns = aq_na_df.columns
        new_columns = [x[:-2] if x.endswith('_x') else x for x in old_columns]

        aq_na_df.columns = new_columns 
        
        new_df2 = pd.merge(aq_na_df,aq_w_dist_group.nth(level),  how='left', left_on=['station_id'], right_on = ['aq_start'])
        aq_na_df_new = pd.merge(new_df2, weather_all[weather_attrs+['station_id',time_column]],  how='left', left_on=['w_end',time_column], right_on = ['station_id',time_column])
        #print(aq_not_na_df.columns)
        #print(aq_na_df_new.columns)
        return pd.concat([aq_not_na_df,aq_na_df_new],axis=0)
		
def fill_weather_gap(aq_df,weather_df,levels,time_column):
    aq_group = get_aq_w_dist()
    aq_all = add_nearest_weather(aq_df,weather_df,aq_group,0,time_column)
    for i in range(1,levels):
        print("Level "+str(i)+" #null before",aq_all[['humidity','pressure','temperature']].isnull().any(axis=1).sum())
        aq_all = add_nearest_weather(aq_all,weather_df,aq_group,i,time_column)
        print("After: ",aq_all[['humidity','pressure','temperature']].isnull().any(axis=1).sum())
    aq_all['station_id'] = aq_all['station_id_x']
    aq_all = aq_all.drop(['station_id_x','station_id_y','aq_start'],axis=1)
    return aq_all

######codes for feature extraction
def get_raw_train_test_data(df,attrs,time_columns,time_key,length=24*7*3):
    '''
        The function prepare the records in the given length for feature extraction and training 
    '''
    grouped = df.groupby('station_id')
    data = []
    for key,group in grouped:
        station_id = group.station_id.values[0]
        station_type_id = group.station_type_id.values[0]
        group = group.sort_values(by=time_key)
        for i in range(0,group.shape[0]-length-48):
            tmp = [station_id,station_type_id]
            #add time info
            tmp += group[time_columns].values[i+length+24].flatten().tolist()
            attrs_raw_in_window = group[attrs].values[i:i+length,:].flatten().tolist()
            tmp += attrs_raw_in_window
            #add the label 
            labels = group[attrs].values[i+length:i+length+48,:].flatten().tolist()
            tmp += labels
            data.append(tmp)
    return data
	
def get_holiday_features(date_columns):
    '''
        Given date_columns ('year','month','day','hour'),return one_hot attributes that following the order
        1.whether it's the one day before holiday
        2.first day of a holiday
        3.last day of holiday
        4.first day of work 
    '''
    holiday_period = [('2017-01-01','2017-01-02'),\
                      ('2017-01-27','2017-02-02'),\
                      ('2017-04-02','2017-04-04'),\
                      ('2017-04-29','2017-05-01'),\
                      ('2017-05-28','2017-05-30'),\
                      ('2017-10-01','2017-10-08'),\
                      ('2017-12-30','2018-01-01'),\
                      ('2018-02-15','2018-02-21'),\
                      ('2018-04-05','2018-04-07'),\
                      ('2018-04-29','2018-05-01'),\
                      ('2018-06-16','2018-06-18'),\
                      ('2018-09-22','2018-08-24'),\
                      ('2018-10-01','2018-10-07')]
    holiday_df = pd.DataFrame(holiday_period,columns=['holi_start','holi_end'])

    holiday_df['holi_start'] = pd.to_datetime(holiday_df['holi_start'])
    holiday_df['holi_end'] = pd.to_datetime(holiday_df['holi_end'])
    
    date_df = pd.DataFrame(date_columns.astype(int),columns=['year','month','day'])
    date_df['date'] = pd.to_datetime(date_df)
    
    onehot_list = []
    for index,row in date_df.iterrows():
        start1 = holiday_df.loc[holiday_df.holi_start == row.date]
        start_neg1 = holiday_df.loc[holiday_df.holi_start == (row.date + timedelta(days=1))]
        last_day = holiday_df.loc[holiday_df.holi_end == row.date]
        workday1 = holiday_df.loc[holiday_df.holi_end == (row.date - timedelta(days=1))] 
        holiday_onehot = [start1.shape[0],start_neg1.shape[0],last_day.shape[0],workday1.shape[0]]
        onehot_list.append(holiday_onehot)
    
    return np.array(onehot_list)

def single_statistical_features(data):
    min_val = data.min(axis=1).reshape(-1,1)
    max_val = data.max(axis=1).reshape(-1,1)
    mean_val = data.mean(axis=1).reshape(-1,1)
    median_val = np.median(data,axis=1).reshape(-1,1)
    std_val = np.std(data,axis=1).reshape(-1,1)
    return np.hstack([min_val,max_val,mean_val,median_val,std_val])

def get_all_statistical_features(data):
    data = data.astype(np.float)
    #get statistic feature across 3 weeks
    stat_all = single_statistical_features(data)
    #get statistical features by day(only for the last week)
    stat_days = []
    for  i in range(14,data.shape[1]//24):
        stat_day = single_statistical_features(data[:,i*24:(i+1)*24])
        stat_days.append(stat_day)
    #get statistical features by week
    stat_weeks = []
    for i in range(0,data.shape[1]//(24*7)):
        stat_week = single_statistical_features(data[:,i*24*7:(i+1)*24*7])
        stat_weeks.append(stat_week)
    return np.hstack([stat_all]+stat_days+stat_weeks)

def one_hot_stations(stations_columns):
    stations_df = pd.DataFrame(stations_columns,columns=['station_name','station_type_id'])
    stations_df['station_type_id'] = stations_df['station_type_id'].apply(int).apply(str)
    stations_df['station_id'] = pd.factorize(stations_df['station_name'])[0]
    stations_df['station_id'] = stations_df['station_id'].apply(int).apply(str)
    one_hot_stations = pd.get_dummies(stations_df[['station_id','station_type_id']],prefix=['station','stationType'])
    return one_hot_stations.values

def get_all_features(data,attr,length):
    stations_features = one_hot_stations(data[:,:2])
    date_features = data[:,2:6]
    holiday_features = get_holiday_features(data[:,2:5])
    if attr.startswith('PM2'):
        stat_features = get_all_statistical_features(data[:,6:6+length])
        raw_features = np.hstack([data[:,6+14*24:6+length],data[:,6+length+18*24:6+length*2]])
    elif attr.startswith('PM10'):
        stat_features = get_all_statistical_features(data[:,6+length:6+length*2])
        raw_features = np.hstack([data[:,6+length+14*24:6+length*2],data[:,6+18*24:6+length]])
    else:
        stat_features = get_all_statistical_features(data[:,6+length*2:])
        raw_features = data[:,6+length*2 + 14*24:]
    return np.hstack([stations_features,date_features,holiday_features,raw_features,stat_features]) 
	
def split_features_labels(data,attr,length=24*7*3):
    X = get_all_features(data[:,:-48*3],attr,length)
    print(attr)
    if attr.startswith('PM2'):
        print("PM2.5")	
        y = data[:,-48*3:-48*2]
    elif attr.startswith('PM10'):
        y = data[:,-48*2:-48]
    else:
        y = data[:,-48:]
    return X,y
	
######prepare data for final training
def get_train_data_final(X,y):
    '''
        One hot the data to be predicted into 48 rows
    '''
    features_num = X.shape[1]
    count = 0
    one_mat = np.ones((48,1))
    train_data = []
    for i in range(X.shape[0]):
        row = X[i,:]
        dup = one_mat*row.T
        #print(y[i,:].reshape(48,-1))
        one_hot = np.eye(48, dtype=int)
        data = np.hstack([dup,one_hot,y[i,:].reshape(48,-1)])
        train_data.append(data)
    return train_data
