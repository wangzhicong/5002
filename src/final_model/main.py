import pandas as pd
import numpy as np
import models
import util
from features_extraction import *



def rbr_predict(models,aq_test,train_df,val_df,attrs_used,attr_pred='PM10'):
    #prepare the pool for rolling
    df_pool = pd.concat([train_df,val_df],axis=0)
    df_pool = df_pool.loc[df_pool.time >= '2018-04-20 00:00:00']
    count = 0
    result_frames = []
    for time_s in aq_test.time.unique():
        curr_row = aq_test.loc[aq_test.time == time_s]
        pool_tmp = pd.concat([df_pool,curr_row],axis=0)
        features_df,attrs_all = add_features_all(pool_tmp,attr_pred)
        test_drop = features_df.drop_duplicates(subset=['station_id','time'])
        test_x = features_df[attrs_used].values
        features_df[attr_pred+'_pred'] = models.predict(test_x)

        features_df = features_df.loc[features_df.time == time_s]
        features_df[attr_pred+'_filled'] = features_df[attr_pred+'_pred']
        #print(df_pool.shape)
        df_pool = df_pool.append(features_df)
        #print(df_pool.shape)

        result_frames.append(features_df)
    return pd.concat(result_frames,axis=0)
	
def results_to_submmision(O3_result,PM25_result,PM10_result,result_all,sample):
    O3_result = O3_result.sort_values(by='time')
    PM10_result = PM10_result.sort_values(by='time')
    PM25_result = PM25_result.sort_values(by='time')
    O3_mean = O3_result.groupby('time')['O3_pred'].mean().values
    PM10_mean = PM10_result.groupby('time')['PM10_pred'].mean().values
    PM25_mean = PM25_result.groupby('time')['PM25_pred'].mean().values
    zhiwuyuan_names = ['zhiwuyuan_aq#'+str(i) for i in range(48)]
    zhiwuyuan_pd = pd.DataFrame({'test_id':zhiwuyuan_names,'PM2.5':PM25_mean,'PM10':PM10_mean,'O3':O3_mean})
    frames = []
    grouped= result_all.groupby('station_id')
    count = 0
    for name,group in grouped:
        group = group.sort_values(by='time')
        group = group.reset_index()
        names = [name+'#'+str(i) for i in range(48)]
        df_tmp = pd.DataFrame({'test_id':names,'PM2.5':group['PM25_pred'].values,'PM10':group['PM10_pred'].values,'O3':group['O3_pred'].values})
        frames.append(df_tmp)

    result_df = pd.concat(frames,axis=0)
    result_df = pd.concat([result_df,zhiwuyuan_pd],axis=0)
    final_result = pd.merge(sample[['test_id']],result_df,how='left',on=['test_id'])
    return final_result
	
if __name__ == '__main__':
	#load historical pollution index and weather data in uniform formate
	aq_2018,aq_2017 = util.load_data_with_weather(start_17='2017-03-01 00:00:00',end_17='2017-05-03 12:00:00',start_18='2018-03-01 00:00:00',end_18='2018-04-01 00:00:00')
	O3_dict,PM10_dict,PM25_dict = models.get_features_model_dict(aq_2017,aq_2018)
	
	print("prediction start")
	O3_result = rbr_predict(**O3_dict)
	print("O3 done")
	PM10_result = rbr_predict(**PM10_dict)
	print("PM10 done")
	PM25_result = rbr_predict(**PM25_dict)
	print("PM25 done")
	
	#save result
	O3_result.to_csv('O3_result.csv')
	PM10_result.to_csv('PM10_result.csv')
	PM25_result.to_csv('PM25_result.csv')

	result_df = O3_result.merge(PM10_result,how='left',on=['station_id','time'])
	result_all = result_df.merge(PM25_result,how='left',on=['station_id','time'])
	result_all.to_csv('all_result.csv')
	
	#convert result to submission format
	sample = pd.read_csv('../data/sample_submission.csv')
	final_result = results_to_submmision(O3_result,PM25_result,PM10_result,result_all,sample)
	final_result.to_csv('submission.csv',index=False)