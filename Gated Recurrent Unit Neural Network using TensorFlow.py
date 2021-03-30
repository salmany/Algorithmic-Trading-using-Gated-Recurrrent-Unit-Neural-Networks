import numpy as np
import pandas_datareader.data as web
import pandas as pd
import sklearn
import sklearn.preprocessing
from matplotlib import pyplot as plt

########### ONLINE STOCKS DATA COLLECTOR
stock_names = ['AAPL', 'AMZN', 'MSFT', 'HMC', 'GOOG', 'TM', 'TSLA', 'UN']
 
on_data = web.DataReader(stock_names,data_source="yahoo",start='1/15/2017', end='1/18/2018')['Adj Close']

on_data.sort_index(inplace=True)
ret_online = on_data.pct_change()
print(ret_online.shape) 
print(ret_online) 

print("Taking data for first stock - AAPL")
print(np.asmatrix(ret_online)[:,0]) 
data = np.asmatrix(ret_online)[:,0]
data2 = data

min_max_scaler = sklearn.preprocessing.MinMaxScaler()
data = min_max_scaler.fit_transform(data.reshape(-1, 1))

print(data)

print("Splitting into Test and Train Data")
train_ratio = 0.8
num_data = int(train_ratio * data.shape[0])

train = data[0:num_data]
test = data[num_data:]

print(num_data)
#print(test)
plt.title('Scaled vs Actual Stock Data')
plt.plot(data, label = "Values", color = 'blue')
plt.plot(data2, label = "Old Values", color = 'green')

plt.show()
