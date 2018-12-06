import pandas as pd
import numpy as np
import datetime


def fill_time_gap(df):
    '''
        Some time is missing for all the stations. This function is used to generate a continuous timeindex to replace the 
        old one for each station. The pollution index for all new generated time will be null
    '''
    delete_date_dict = {}
    grouped = df.groupby('station_id')
    frames = []
    for name,group in grouped:
        group = group.sort_values(by='time')
        #group = group.drop_duplicates(subset=['time'])
        time_min = group.time.unique().min()
        time_max = group.time.unique().max()
        idx = pd.date_range(time_min, time_max, freq="1h")
        fill_date = idx[~idx.isin(group.time.values)]
        delete_date_dict[name] = fill_date
        group = group.reindex(idx)
        group['time'] = idx.strftime('%Y-%m-%d %H:00:00')
        reused_cols = ['day', 'dist', 'hour','latitude', 'longitude', 'month', 'station_id',
       'station_id_int', 'station_type_id', 'time', 'timestamp','w_end','year']
        group[reused_cols] = group[reused_cols].fillna(method='ffill')
        frames.append(group)
    return pd.concat(frames,axis=0).reset_index(drop=True),delete_date_dict

def fill_weather_by_mode(df):
    '''
        This function is used to fill yje missing weather attributes by the mode of weather among all the stations.
    '''
    df = df.sort_index()
    df['weather_fill'] = df.groupby('time')['weather'].apply(lambda x:x.fillna(x.mode().values[0] if x.mode().shape[0] > 0 else np.nan) ).sort_index()
    df['weather_fill'] = df['weather_fill'].fillna(method='ffill')
    return df

def get_rolling_fill(df,attrs):
    groups = df.groupby('station_id')
    df = None
    for key,group in groups:
        group = group.sort_values(by='time')
        for attr in attrs:
            group[attr+'_filled'] = group[attr+'_filled'].rolling(5,center = True, min_periods = 1).median()
        try:
            df =df.append(group)
        except:
            df = group
            
    return df

def cut_outliers(df,target,threshold):
    for i in target:
        df[i] = df[i].apply(lambda x: x if x <= threshold else threshold)
    
    return df

def fill_null_attrs(df,attrs):
    
    df = df.sort_index()
    for attr in attrs:
        df[attr+'_filled'] = df.groupby('time')[attr].apply(lambda x: x.fillna(x.mean())).sort_index()
   
    return df


#######Codes for features extraction
def one_hot_stations(stations_df):
    stations_df['station_type_id'] = stations_df['station_type_id'].apply(int).apply(str)
    stations_df['station_id_int'] = stations_df['station_id_int'].apply(int).apply(str)
    stations = pd.get_dummies(stations_df[['station_id_int','station_type_id']],prefix=['station','stationType'])
    return pd.concat([stations_df,stations],axis=1),stations.columns

def add_previous(df,attr,hours=5,neg = 1):
    '''
        add the raw values in the previous days
    '''
    for i in range(1,hours+1):
        if neg > 0:
            df[attr+"_"+str(i)+ "_shift"] =df[attr].shift(i)
        else:
            df[attr+"_"+str(i)+ "_shift"] = -df[attr].shift(i)
    new_attrs = [attr+"_"+str(i)+ "_shift" for i in range(1,hours+1)]
    return df,new_attrs

def add_rolling_PM25(df,attrs,freqs=['3h','6h','12h','1D','3D','7D'],neg = 1):
    df = df.sort_values(by='time')
    df.index = pd.to_datetime(df.time)
    new_attrs = []
    for attr in attrs:
        ##calculate the shift
        df[attr] = df[attr].fillna(method='ffill')
        df[attr+'shift1'] = df[attr].shift(1)
        df[attr+'_diff'] = df[attr] - df[attr+'shift1']
        df[attr+'_diff'] = df[attr+'_diff'].shift(1)
        new_attrs.append(attr+'_diff')
        for freq_str in freqs:
            rolling_df = df[attr].rolling(freq_str)
            if neg > 0:
                df[attr+'_'+freq_str+'_std'] = rolling_df.var().shift(1)
                df[attr+'_'+freq_str+'_median'] = rolling_df.median().shift(1)
            else:
                df[attr+'_'+freq_str+'_std'] = -rolling_df.var().shift(1)
                df[attr+'_'+freq_str+'_median'] = -rolling_df.median().shift(1)
            stat_attrs = [attr+'_'+freq_str+x for x in ['_std','_median'] ]#['_mean','_min','_max','_sum','_std','_median']
            new_attrs += stat_attrs
    return df,new_attrs

def add_rolling_O3(df,attrs,freqs=['3h','6h','12h','1D','3D','7D']):
    df = df.sort_values(by='time')
    df.index = pd.to_datetime(df.time)
    new_attrs = []
    for attr in attrs:
        ##calculate the shift
        df[attr] = df[attr].fillna(method='ffill')
        df[attr+'shift1'] = df[attr].shift(1)
        df[attr+'_diff'] = df[attr] - df[attr+'shift1']
        df[attr+'_diff'] = df[attr+'_diff'].shift(1)
        new_attrs.append(attr+'_diff')
        for freq_str in freqs:
            rolling_df = df[attr].rolling(freq_str)
            df[attr+'_'+freq_str+'_mean'] = rolling_df.mean().shift(1)
            df[attr+'_'+freq_str+'_min'] = rolling_df.min().shift(1)
            df[attr+'_'+freq_str+'_max'] = rolling_df.max().shift(1)
            df[attr+'_'+freq_str+'_sum'] = rolling_df.sum().shift(1)
            df[attr+'_'+freq_str+'_std'] = rolling_df.var().shift(1)
        stat_attrs = [attr+'_'+freq_str+x for x in ['_mean','_min','_max','_sum','_std']]
        new_attrs += stat_attrs
    return df,new_attrs

def add_rolling_PM10(df,attrs,freqs=['3h','6h','12h','1D','3D','7D']):
    df = df.sort_values(by='time')
    df.index = pd.to_datetime(df.time)
    new_attrs = []
    for attr in attrs:
        ##calculate the shift
        df[attr] = df[attr].fillna(method='ffill')
        for i in range(1,7):
            df[attr+'shift'+str(i)] = df[attr].shift(i)
        df[attr+'_diff'] = df[attr] - df[attr+'shift1']
        df[attr+'_diff12'] = df[attr+"shift1"] - df[attr+"shift2"]
        df[attr+'_diff23'] = df[attr+"shift2"] - df[attr+"shift3"]
        df[attr+'_diff34'] = df[attr+"shift3"] - df[attr+"shift4"]
        df[attr+'_diff45'] = df[attr+"shift4"] - df[attr+"shift5"]
        
        diff_attrs = [attr+"_diff"+str(i)+str(i+1) for i in range(1,4)]
        df[attr+"_diff_mean"] = df[diff_attrs].mean(axis=1)
        df[attr+"_diff_sum"] = df[diff_attrs].sum(axis=1)
        df[attr+"_diff_var"] = df[diff_attrs].var(axis=1)
        
        
        df[attr+'_diff'] = df[attr+'_diff'].shift(1)
        new_attrs+=[attr+"_diff_mean",attr+"_diff_sum",attr+"_diff_var"]
        for freq_str in freqs:
            rolling_df = df[attr].rolling(freq_str)
            df[attr+'_'+freq_str+'_mean'] = rolling_df.mean().shift(1)
            '''df[attr+'_'+freq_str+'_min'] = rolling_df.min().shift(1)
            df[attr+'_'+freq_str+'_max'] = rolling_df.max().shift(1)'''
            df[attr+'_'+freq_str+'_sum'] = rolling_df.sum().shift(1)
            df[attr+'_'+freq_str+'_std'] = rolling_df.var().shift(1)
        stat_attrs = [attr+'_'+freq_str+x for x in ['_mean','_sum','_std']]
        new_attrs += stat_attrs
    return df,new_attrs

def add_pm10_features_single(group):
    '''
        get all the features needed to predict PM10
    '''
    #attrs for PM10
    group.index = group['timestamp']
    new_attrs_all = []
    group,new_attrs_tmp = add_rolling_PM10(group,['PM10_filled'],['3h','6h','12h','1D','3D'])
    new_attrs_all += new_attrs_tmp
    group,new_attrs_tmp = add_previous(group,'PM10_filled',6)
    group['PM10_filled_diff12'] = group['PM10_filled_1_shift']

    
    
    
    #same last few days values at the same hour
    group['PM10_1d_shift'] = group['PM10_filled'].shift(24)
    group['PM10_2d_shift'] = group['PM10_filled'].shift(48)
    group['PM10_1d_shift1'] = group['PM10_filled']
    new_attrs_all += ['PM10_1d_shift','PM10_2d_shift']
    
    
    #weather_attrs
    weather_used = ['pressure','temperature','humidity','wind_speed']
    group,new_attrs_tmp = add_rolling_PM10(group,weather_used,['6h','12h','1D'])
    new_attrs_all += new_attrs_tmp
    for w in weather_used:
        group,new_attrs_tmp = add_previous(group,w,3)
        new_attrs_all += new_attrs_tmp
        
    #add bad_weather sum
    group['bad_weather_6hsum'] = group['bad_weather'].rolling('6h').sum()
    #group['bad_weather_1dsum'] = group['bad_weather'].rolling('1D').sum()
    new_attrs_all += ['bad_weather_6hsum']
    
    #add holidays features
    holiday_period = [[datetime.date(2018, 4, 5),datetime.date(2018, 4, 7)],\
                     [datetime.date(2018, 5, 1),datetime.date(2018, 5, 2)]]
    group['holiday'] = 0
    for holi_p in holiday_period:
        group[holi_p[0]:holi_p[1]]['holiday'] = 1
    group['holiday_sum'] = group['holiday'].rolling('1D').sum()
    new_attrs_all += ['holiday_sum','holiday']
    

    
    
    min_date = group.index.min()
    start_date = min_date+datetime.timedelta(days=7)
    group = group[start_date:]
    return group,new_attrs_all

def add_O3_features_single(group):
    '''
        get all the features needed to predict O3 in a single station
    '''
    #attrs for O3
    group.index = group['timestamp']
    new_attrs_all = []
    group,new_attrs_tmp = add_rolling_O3(group,['O3_filled'],['3h','6h','12h','1D'])
    new_attrs_all += new_attrs_tmp
    
    #same last few days values at the same hour, look at seasonal property
    group['O3_1d_shift'] = group['O3_filled'].shift(24)
    group['O3_2d_shift'] = group['O3_filled'].shift(48)
    group['O3_3d_shift'] = group['O3_filled'].shift(24*3)
    group['O3_4d_shift'] = group['O3_filled'].shift(24*4)
    group['O3_5d_shift'] = group['O3_filled'].shift(24*5)
    shift_cols = ['O3_'+str(i)+'d_shift' for i in range(1,4)]
    
    group['O3_shift_sum'] = group[shift_cols].sum(axis=1)
    group['O3_shift_mean'] = group[shift_cols].mean(axis=1)
    group['O3_shift_var'] = group[shift_cols].var(axis=1)
    
    group['O3_12_diff'] = group['O3_1d_shift'] - group['O3_2d_shift']
    group['O3_23_diff'] = group['O3_2d_shift'] - group['O3_3d_shift']
    group['O3_34_diff'] = group['O3_3d_shift'] - group['O3_4d_shift']
    day_diff_cols = ['O3_12_diff','O3_23_diff','O3_34_diff']
    group['O3_diff_sum'] = group[day_diff_cols].sum(axis=1)
    group['O3_diff_var'] = group[day_diff_cols].var(axis=1)
    group['O3_diff_mean'] = group[day_diff_cols].mean(axis=1)
    #group['O3_season_shift'] = group['O3_filled'].shift(230)
    new_attrs_all += day_diff_cols
    new_attrs_all += shift_cols
    new_attrs_all += ['O3_diff_'+x for x in ['sum','var','mean']]
    new_attrs_all += ['O3_shift_'+x for x in ['sum','var','mean']]
    
    #weather_attrs
    weather_used = ['pressure','temperature','humidity','wind_speed']
    group,new_attrs_tmp = add_rolling_O3(group,weather_used,['6h','12h','1D','3D'])
    new_attrs_all += new_attrs_tmp

        
    #add bad_weather sum
    group['bad_weather_6hsum'] = group['bad_weather'].rolling('6h').sum()
    new_attrs_all += ['bad_weather_6hsum']
    
    #add holidays features
    holiday_period = [[datetime.date(2018, 4, 5),datetime.date(2018, 4, 7)],\
                     [datetime.date(2018, 5, 1),datetime.date(2018, 5, 2)],\
                     [datetime.date(2017,4,2),datetime.date(2017,4,4)],\
                     [datetime.date(2017,4,29),datetime.date(2017,5,1)]]
    group['holiday'] = 0
    for holi_p in holiday_period:
        group[holi_p[0]:holi_p[1]]['holiday'] = 1
    group['holiday_sum'] = group['holiday'].rolling('1D').sum()
    new_attrs_all += ['holiday_sum','holiday']
    

    
    
    min_date = group.index.min()
    start_date = min_date+datetime.timedelta(days=7)
    group = group[start_date:]
    return group,new_attrs_all
    

def add_pm25_features_single(group):
    '''
        get all the features needed to predict PM25
    '''
    group.index = group['timestamp']
    new_attrs_all = []
    
    #PM2.5 attrs wow not use PM2.5 values can have better evaludation result on validation set
    group,new_attrs_tmp = add_rolling_PM25(group,['PM25_filled'],['6h'])
    new_attrs_all += new_attrs_tmp
    group,new_attrs_tmp = add_previous(group,'PM25_filled',2)
    new_attrs_all += new_attrs_tmp
    
    weather_used_pos = ['humidity']
    weather_used_neg = ['wind_speed']
    group,new_attrs_tmp = add_rolling_PM25(group,weather_used_pos,['6h'],neg=1)
    new_attrs_all += new_attrs_tmp
    group,new_attrs_tmp = add_rolling_PM25(group,weather_used_neg,['6h'],neg=-1)
    new_attrs_all += new_attrs_tmp

    #new_attrs_all += new_attrs_tmp
    for w in weather_used_pos:
        group,new_attrs_tmp = add_previous(group,w,2)
        new_attrs_all += new_attrs_tmp
    for w in weather_used_neg:
        group,new_attrs_tmp = add_previous(group,w,2,neg=-1)
        new_attrs_all += new_attrs_tmp
        
    #add bad_weather sum
    group['bad_weather_6hsum'] = group['bad_weather'].rolling('6h').sum()
    group['bad_weather'] = group['bad_weather']
    new_attrs_all += ['bad_weather_6hsum']
    
    #add holidays features
    min_date = group.index.min()
    start_date = min_date+datetime.timedelta(days=7)
    group = group[start_date:]
    return group,new_attrs_all

def add_features_all(df,attr='PM10'):
    grouped = df.groupby('station_id')
    frames = []
    for name,df_group in grouped:
        #print(name,df_group.time.unique()[-10:])
        if attr == 'PM10':
            #print("Generating features for PM10")
            group,new_features = add_pm10_features_single(df_group)
        elif attr == 'O3':
            #print("Generating features for O3")
            group, new_features = add_O3_features_single(df_group)
        else:
            #print("Generating features for PM2.5")
            group,new_features = add_pm25_features_single(df_group)
        group = group.sort_index().reset_index(drop=True)
        frames.append(group)
    return pd.concat(frames,axis=0),new_features