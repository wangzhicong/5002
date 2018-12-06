
# coding: utf-8

# In[1]:


from keras.layers.core import Dense, Activation, Dropout 
from keras.models import Sequential 
import time 
import math 
from keras.layers import LSTM 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import mean_squared_error 
from keras import optimizers 
import os
from keras.optimizers import SGD,Adam
from keras.layers.normalization import BatchNormalization as bn
import evaluation


# main network definition
def train_all_lstm(x,y,epoch=200,seq = True,time_stamp = 10,unit = 128 ,dropout=0.0,verbose=0,loss ='mse'):
    attr_num = x.shape[1]
    train_X,train_y=x,y
    train_X= train_X.reshape(-1,1,attr_num)
    train_y = train_y.reshape(-1,3)
    trainset={'x':[],'y':[]}
    
    import random
    for i in range(train_X.shape[0]-time_stamp+1):
        trainset['x'].append((train_X[i:i+time_stamp,:,:]).reshape(-1,time_stamp,attr_num))
        trainset['y'].append((train_y[i:i+time_stamp,:]).reshape(-1,time_stamp,3))
    trainset['x'] = np.array(trainset['x']).reshape(-1,time_stamp,attr_num)

    trainset['y'] = np.array(trainset['y']).reshape(-1,time_stamp,3)
    
    model = Sequential()
    model.add(LSTM(unit,return_sequences=True,dropout=dropout))
    #model.add(bn())
    model.add(LSTM(unit,return_sequences=seq,dropout=dropout))
    #model.add(bn())
    #model.add(LSTM(unit,return_sequences=seq,dropout=dropout))
    #model.add(bn())
    model.add(Dense(unit//4,activation='relu'))
    model.add(bn())
    model.add(Dense(3,activation='relu'))
    model.compile(loss=loss, optimizer='adam')
    # fit network
    #[:,-1,:].reshape(-1,3)
    if not seq:
        history = model.fit(trainset['x'], trainset['y'][:,-1,:].reshape(-1,1), epochs=epoch, batch_size=32,  verbose=verbose, shuffle=True)
    else:
        history = model.fit(trainset['x'], trainset['y'], epochs=epoch, batch_size=64,  verbose=verbose, shuffle=True,validation_split =0.1)
    
    return history,model



# split train val
def get_data(train_X,train_y,trainset,valset):   
    import random
    for i in range(train_X.shape[0]-time_stamp+1):
        if random.random()<0.9:
            trainset['x'].append((train_X[i:i+time_stamp,:,:]).reshape(-1,time_stamp,attr_num))
            trainset['y'].append((train_y[i:i+time_stamp,:]).reshape(-1,time_stamp,3))
        else:
            valset['x'].append((train_X[i:i+time_stamp,:,:]).reshape(-1,time_stamp,attr_num))
            valset['y'].append((train_y[i:i+time_stamp,:]).reshape(-1,time_stamp,3))
    trainset['x'] = np.array(trainset['x']).reshape(-1,time_stamp,attr_num)
    valset['x'] = np.array(valset['x']).reshape(-1,time_stamp,attr_num)
    trainset['y'] = np.array(trainset['y']).reshape(-1,time_stamp,3)
    valset['y'] = np.array(valset['y']).reshape(-1,time_stamp,3)



# complete the time line
def complete_time(dataframe):
    #dataframe['time'] = pd.to_datetime(dataframe['time'])
    #print(dataframe.head())
    stations = dataframe['station_id'].unique()
    whole=None
    for i in list(stations):
        #print(i)
        section = pd.DataFrame({'time':pd.date_range(start='2018-04-01 00:00:00', end='2018-04-30 23:00:00', freq='H')})
        section['time']=section['time'].apply(lambda x: str(x))
        section = section.merge(dataframe[dataframe['station_id'] == i],how='left',on=['time'])
        section['station_id'] = section['station_id'].fillna(i)
        try:
            whole = whole.append(section)
        except:
            whole = section
        #print(whole)
    return whole.sort_values(by=['time'])


# In[5]:


import numpy as np 
from matplotlib import pyplot 
import pandas as pd 


airq = pd.read_csv('./aiqQuality_201804.csv')
grid = pd.read_csv('./gridWeather_201804.csv')
observe = pd.read_csv('./observedWeather_201804.csv')
test_grid = pd.read_csv('gridWeather_20180501-20180502.csv')
test_observe = pd.read_csv('observedWeather_20180501-20180502.csv')
airq = complete_time(airq)

airq.fillna(method='ffill')
airq_5= generate(airq)

grid =grid.append(test_grid)
grid2 = clean_weather(grid)


observe = observe.append(test_observe)
observe2 = clean_weather(observe)

airq2= airq.append(airq_5)



def cut_time(data,name,start,end):
    data['utc_time'] = pd.to_datetime(data['utc_time'])
    len(airq)
    df = data.set_index('utc_time')
    df = df.truncate(before=start)
    df = df.truncate(after=end)
    return df


import preprocess
after_data = get_weather_fill(airq2,grid,1,'time')


# In[9]:


dataset = None
groups = airq2.groupby('time')



#del the unnecessary attributes
del after_data['id']
del after_data['CO_Concentration']
del after_data['NO2_Concentration']
del after_data['SO2_Concentration']
after_data = after_data.sort_values(by='time').reset_index(drop=True)


# In[11]:


# predict with prediction as attribute
def pred_lstm(init_x,x,model,scaler,period=47):
    output = None
    timestamp,attr = init_x.shape
    input_x = init_x.reshape(1,-1,attr)
    preds = model.predict(input_x)
    preds = preds.reshape(-1,1)
    tmp = preds[-1,].reshape(-1)
    output = tmp
    #tmp = scaler.transform(tmp.reshape(-1,3))
    for i in range(period):
        #print(preds[-1,],preds.shape)
        
        #print(x[i,0:3])
        tmp_x = list(tmp.reshape(-1)) + list(x[i,3:])
        #print(tmp_x.shape,tmp.shape)
        tmp_all = np.array(tmp_x).reshape(-1,1,attr)#np.append(tmp,tmp_x,axis=0).reshape(-1,1,attr)
        #print(tmp,tmp_all)
        #print(tmp_all.shape,input_x.shape)
        input_x = np.append(input_x,tmp_all,axis=1)
        input_x = input_x[:,-timestamp:,:]
        
        preds = model.predict(input_x)
        preds = preds.reshape(-1,1)
        tmp = preds[-1,].reshape(-1)
        
        output = np.vstack((output,tmp))
        #tmp = scaler.transform(tmp.reshape(-1,3))

    
    return output
    
#prediction with target are not attributes
def pred_lstm_nopred(x,model,time_stamp,period=48):
    output = None
    _,attr = x.shape

    for i in range(period):
        preds = model.predict(x[i:i+time_stamp,:].reshape(1,time_stamp,attr))
        preds = preds.reshape(-1,3)
        #print(preds[-1,],preds.shape)
        tmp = preds[-1,].reshape(-1)
        try:
            output = np.vstack((output,tmp))
        except:
            output = tmp
    
    return output
    
    



#training
from keras.models import load_model 
import sklearn 

stations = list(after_data['station_id'].unique())
his = {}
scores = {}
seq = not False
unit = 64
dropout=0.1
time_stamp = 10
epoch = 300
load = False
results = {}
loss = 'mse'
groups = after_data.groupby('station_id')
for i,group in groups:
    print(i)
    tmp = group
    tmp = tmp.fillna(method ='bfill')
    #tmp['time'] = tmp['time'].apply(lambda x: pd.to_datetime(x))
    #print(group.tail())
    
    #t = tmp['weather'].values.reshape(-1,1)
    #del tmp['weather']
    del tmp['station_id']
    del tmp['time']
    
    #print(tmp.tail(48))
    encoder = sklearn.preprocessing.LabelBinarizer()
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler2 = StandardScaler()
    x = tmp.values
    y = x[:,0:3]
    x[:,3:] = scaler.fit_transform(x[:,3:])
    
    print(x.shape)
    train_x = x[:-49,]
    train_y = y[1:-48]
    test_x = x[-47:,]
    init_test_x = x[-47-time_stamp:-47,:]
    label = y[-48:,]
    
    #print('trainging',i)
    if not load:
        his[i],model= train_all_lstm(train_x,train_y,epoch=epoch,seq=seq,time_stamp = time_stamp,unit=unit,dropout=dropout,verbose=2,loss=loss)
        model.save('models/'+i+'.h5')
    else:
        model= load_model('models/'+i+'.h5')

    



import matplotlib.pyplot as plt
plt.figure()
a = list(prediction.reshape(-1,3)[:,0])
b = list(gts[i].reshape(-1,3)[:,0])
plt.figure()
plt.plot(a,'b')
plt.plot(b,'r')
plt.show()

