import pandas as pd
import numpy as np
import geopy.distance

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


    aq_group = aq_w_dist_df.groupby('aq_start',as_index=False)
    return aq_group

def fill_small_gap(df,groupKey,attr):
    grouped = df.groupby(groupKey)
    frames = []
    for key,group in grouped:
        group = group.sort_values(by='time')
        X = group[attr].values
        consecutive = 0
        for i in range(1,X.shape[0]):
            if pd.isnull(X[i]):
                consecutive += 1
            else:
                if consecutive < 4:
                    start_idx = i - consecutive - 1
                    if start_idx >= 0 and (not pd.isnull(X[start_idx])):
                        start = X[start_idx]
                        end = X[i]
                        for j in range(1,consecutive+1):
                            X[i-j] = end * (1-j/(consecutive+1) ) + start*j/(consecutive+1)
                            if pd.isnull(X[i-j]):
                                print("set up fail")
                consecutive = 0 #reset the counter
        group[attr] = X
        frames.append(group.copy())
    return pd.concat(frames)

def add_nearest_weather(aq_df,weather_all,aq_w_dist_group,level):
    '''
        This function find the nearesr weather station for the air quality data
    '''
    if level == 0:
        new_df1 = pd.merge(aq_df, aq_w_dist_group.nth(0),  how='left', left_on=['station_id'], right_on = ['aq_start'])
        aq_all = pd.merge(new_df1, weather_all,  how='left', left_on=['w_end','time'], right_on = ['station_id','time'])
        return aq_all
    else:
        aq_na_df = aq_df[aq_df[['humidity','pressure','temperature','weather','wind_direction','wind_speed']].isnull().any(axis=1)]
        aq_not_na_df = aq_df[aq_df[['humidity','pressure','temperature','weather','wind_direction','wind_speed']].notnull().any(axis=1)]
        aq_na_df = aq_na_df.drop(['aq_start','dist', 'w_end','humidity', 'id_y', 'pressure', 'station_id_y', 'temperature','weather', 'wind_direction', 'wind_speed'],axis=1)
        old_columns = aq_na_df.columns
        new_columns = [x[:-2] if x.endswith('_x') else x for x in old_columns]

        aq_na_df.columns = new_columns

        new_df2 = pd.merge(aq_na_df,  aq_w_dist_group.nth(level),  how='left', left_on=['station_id'], right_on = ['aq_start'])
        aq_na_df_new = pd.merge(new_df2, weather_all,  how='left', left_on=['w_end','time'], right_on = ['station_id','time'])
        return pd.concat([aq_not_na_df,aq_na_df_new])

def add_nearby_weather_mean(aq_df,weather_all,aq_w_dist_group,level):
    '''
        Use the mean of weather station that not close enough to fill_na_values if there is not close weather data found
    '''
    aq_na_df = aq_df[aq_df[['humidity','pressure','temperature','weather','wind_direction','wind_speed']].isnull().any(axis=1)]
    aq_not_na_df = aq_df[aq_df[['humidity','pressure','temperature','weather','wind_direction','wind_speed']].notnull().any(axis=1)]
    aq_na_df = aq_na_df.drop(['aq_start','dist', 'w_end','humidity', 'id_y', 'pressure', 'station_id_y', 'temperature','weather', 'wind_direction', 'wind_speed'],axis=1)
    old_columns = aq_na_df.columns
    new_columns = [x[:-2] if x.endswith('_x') else x for x in old_columns]
    aq_na_df.columns = new_columns

    _aq_group  = aq_na_df.groupby(['station_id','time'])
    frames = []
    not_found = []
    for key,group in _aq_group:
        _station_id = group.station_id.values[0]
        _time = group.time.values[0]
        _nearbyStationGroup = aq_w_dist_group.head(level)
        _nearbyStations = _nearbyStationGroup.loc[_nearbyStationGroup.aq_start == _station_id]
        dist_mean = _nearbyStations.dist.mean()
        weather_around_df = weather_all.loc[(weather_all.time == _time) & (weather_all.station_id.isin(_nearbyStations.w_end.values))].dropna()
        if weather_around_df.shape[0] > 0:
            weather_around = weather_around_df.mean().to_dict()
            weather_around['aq_start'] = _station_id
            weather_around['dist'] = dist_mean
            weather_around['w_end'] = 'top'+str(level)+'_mean'
            weather_around['time'] = _time
            weather_around['weather'] = weather_around_df.weather.mode()
            mean_df = pd.DataFrame(weather_around,index=[0])

            group_new = pd.merge(group, mean_df,  how='left', left_on=['station_id','time'], right_on = ['aq_start','time'])
            group_new['station_id_x'] = group_new['station_id']
            frames.append(group_new)
        else:
            group['station_id_x'] = group['station_id']
            not_found.append(group)
    result_df = pd.concat([aq_not_na_df]+frames+not_found)
    result_df = result_df.drop(['station_id'],axis=1)
    return result_df

def get_rolling_static(df,target_attr,group_key,win_size):
    grouped = df.groupby(group_key)
    frames = []
    for key,group in grouped:
        rolling_obj = group[target_attr].rolling(win_size,min_periods=1)
        prefix = target_attr+"win_"+str(win_size)
        group[prefix+'_mean'] = rolling_obj.mean()
        group[prefix+'_median'] = rolling_obj.median()
        group[prefix+'_sum'] = rolling_obj.sum()
        group[prefix+'_min'] = rolling_obj.min()
        group[prefix+'_max'] = rolling_obj.max()
        group[prefix+'_std'] = rolling_obj.var()
        statistic_columns1 = ['mean','median','sum','min','max','std']
        statistic_columns = [prefix+'_'+x for x in statistic_columns1]
        #have to shift the values or the features will contain the information about the label
        group[statistic_columns] = group[statistic_columns].shift(1)
        frames.append(group)
    return pd.concat(frames),statistic_columns

def preprocess(aq_df,weather_df,gridw_df):
	#linear fill the small gap only(gap larger than 4 use ffill)
	aq_df = fill_small_gap(aq_df,'station_id','PM25_Concentration')
	aq_df = fill_small_gap(aq_df,'station_id','PM10_Concentration')
	aq_df = fill_small_gap(aq_df,'station_id','NO2_Concentration')
	aq_df = fill_small_gap(aq_df,'station_id','CO_Concentration')
	aq_df = fill_small_gap(aq_df,'station_id','O3_Concentration')
	aq_df = fill_small_gap(aq_df,'station_id','SO2_Concentration')
	aq_df = aq_df.fillna(method="ffill")


	#delete the abnormal weather data
	weather_df = weather_df.loc[weather_df.temperature < 100]

	#add weather
	#step1 add the top5 nearest weather data(only go to the next step if not found)
	weather_all = pd.concat([weather_df,gridw_df])
	aq_group = get_aq_w_dist()
	aq_all_df = add_nearest_weather(aq_df,weather_all,aq_group,0)
	for i in range(1,5):
		print("Level "+str(i)+" #null before",aq_all_df[['humidity','pressure','temperature','weather','wind_direction','wind_speed']].isnull().any(axis=1).sum())
		aq_all_df = add_nearest_weather(aq_all_df,weather_all,aq_group,i)
		print("After: ",aq_all_df[['humidity','pressure','temperature','weather','wind_direction','wind_speed']].isnull().any(axis=1).sum())
	#step2 use nearby weather data at given time
	aq_all_df = add_nearby_weather_mean(aq_all_df,weather_all,aq_group,26)
	aq_all_df = aq_all_df.fillna(method="ffill")

	#rename the columns after join
	old_columns = aq_all_df.columns
	new_columns = [x[:-2] if x.endswith('_x') else x for x in old_columns]
	aq_all_df.columns = new_columns
	aq_all_df = aq_all_df.drop(['id_y','station_id_y','w_end','aq_start'],axis=1)

	#optional can comment this part, statistical info of the past data

	return aq_all_df


'''
	win_size_list = [72,36,24,12,6]
	stat_cols = []
	for win_size in win_size_list:
		aq_all_df,new_cols1 = get_rolling_static(aq_all_df,'PM25_Concentration','station_id',win_size)
		aq_all_df,new_cols2 = get_rolling_static(aq_all_df,'PM10_Concentration','station_id',win_size)
		aq_all_df,new_cols3 = get_rolling_static(aq_all_df,'O3_Concentration','station_id',win_size)
		stat_cols = stat_cols + new_cols1 + new_cols2 + new_cols3
'''
