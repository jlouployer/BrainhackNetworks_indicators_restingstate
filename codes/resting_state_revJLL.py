# Pouya Ghaemmaghami
# coding: utf-8

# In[89]:


# import necessary packages:
import pandas as pd
import numpy as np
import scipy
import math
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from __future__ import division



import numpy as np
import matplotlib.pyplot as plt
from pandas import datetime
import math
from sklearn import preprocessing
import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
#import pandas_datareader.data as web
from scipy import stats

# In[103]:


# define the indicators:
def moving_average(x,N): 
    """ return moving average from last N samples: """ 
    SMA = 0
    for i in range(N):
        SMA = SMA + (1/N)*x[-i-1]
    return SMA

def exponential_weighted_moving_average(x,N,alpha): 
    """ return exponential weighted moving average from last N samples: """ 
    EWMA = 0
    s = [x[-1]]
    for i in range(N):
        if i == 0:
            s_i = x[-1]
        else:
            s_i = alpha*x[-i-1] + (1 - alpha)*s[i-1]
        EWMA = EWMA + (1/N)*s_i
        s.append(s_i)
    return EWMA

def relative_strength_index(x,N,alpha):
    """ return relative strength index from the last N samples: """
    ups = np.zeros((N,1))
    downs = np.zeros((N,1))
    for ind in range(N):
        if x[-1-ind] > x[-1-ind-1]:
            ups[ind,0] = x[-1-ind] - x[-1-ind-1]
        if x[-1-ind] < x[-1-ind-1]:
            downs[ind,0] = x[-1-ind-1] - x[-1-ind]
    
    EWMA_ups = 0
    s = [ups[-1]]
    for i in range(N):
        if i == 0:
            s_i = s[-1]
        else:
            s_i = alpha*ups[-i-1] + (1 - alpha)*s[i-1]
        EWMA_ups = EWMA_ups + (1/N)*s_i
        s.append(s_i)
        
    EWMA_downs = 0
    s = [downs[-1]]
    for i in range(N):
        if i == 0:
            s_i = downs[-1]
        else:
            s_i = alpha*downs[-i-1] + (1 - alpha)*s[i-1]
        EWMA_downs = EWMA_downs + (1/N)*s_i
        s.append(s_i)
    
    RS = EWMA_ups/EWMA_downs
    RSI = 100 - (100/(1 + RS))
    return RSI

def Bollinger_bands(x,N,alpha,K):
    """ return Bollinger bands: """ 
    # SMA = 0
    # for i in range(N):
    #    SMA = SMA + (1/N)*x[-i-1]
    SMA = np.mean(x[-N:-1])   
    upper_band = SMA + K*np.std(x[-N:-1])   
    lower_band = SMA - K*np.std(x[-N:-1])   
    return SMA, upper_band, lower_band
    


# In[91]:


# load the data:
df     = pd.read_csv('../datasets/fMRI/day1/100307.csv',header=None)
data   = df.values
Nvars  = data.shape[1]
labels_Glasser  = pd.read_csv('../datasets/fMRI/labels_Glasser.csv',header=None)[0].tolist()
df.columns = labels_Glasser
TR     = 0.72 #[s]


# In[92]:


# select one time series:
ROI_number = 0
x = data[:,ROI_number]
ROI_name = labels_Glasser[ROI_number]
N = len(x)


# In[111]:


# calculate the indicators:
# simple moving average:
N_SMA = 14
SMAvec = np.zeros((N - N_SMA,1))
for ind in range(len(SMAvec)):
    SMAvec[ind] = moving_average(x[:-1-ind],N_SMA)
    
# exponential weighted moving average:    
N_EWMA = 14
alpha = 0.2
EWMAvec = np.zeros((N - N_EWMA,1))
for ind in range(len(EWMAvec)):
    EWMAvec[ind] = exponential_weighted_moving_average(x[:-1-ind],N_SMA,alpha)

# MACD:
alpha = 0.2
N_EWMA1 = 12
EWMAvec1 = np.zeros((N - N_EWMA1,1))
for ind in range(len(EWMAvec1)):
    EWMAvec1[ind] = exponential_weighted_moving_average(x[:-1-ind],N_EWMA1,alpha)
N_EWMA2 = 26
EWMAvec2 = np.zeros((N - N_EWMA2,1))
for ind in range(len(EWMAvec2)):
    EWMAvec2[ind] = exponential_weighted_moving_average(x[:-1-ind],N_EWMA2,alpha)
    
# RSI:
alpha = 0.2
N_RSI = 14
RSIvec = np.zeros((N - N_RSI - 1,1))
for ind in range(len(RSIvec)):
    RSIvec[ind] = relative_strength_index(x[:-1-ind],N_RSI,alpha)
    
# Bollinger bands:
K = 1.0
alpha = 0.2
N_BOLL = 14
BOLLvec_ma = np.zeros((N - N_BOLL,1))
BOLLvec_upper = np.zeros((N - N_BOLL,1))
BOLLvec_lower = np.zeros((N - N_BOLL,1))
for ind in range(len(BOLLvec_ma)):
    Boll = Bollinger_bands(x[:-1-ind],N_BOLL,alpha,K)
    BOLLvec_ma[ind] = Boll[0]
    BOLLvec_upper[ind] = Boll[1]
    BOLLvec_lower[ind] = Boll[2]


# In[112]:


# plot simple and exponential moving average:
plt.figure(figsize=(20,8))
plt.plot(np.arange(N)*TR, x, color='k', label='data')
plt.plot(np.arange(N)[N_SMA:]*TR, SMAvec, color='r', label='SMA')
plt.plot(np.arange(N)[N_EWMA:]*TR, EWMAvec, color='g', label='EWMA')
plt.legend()
plt.xlabel('time')
plt.ylabel('BOLD response')
plt.xlim([0, N*TR*0.25]) #zoom inot first 25% of the chart
plt.title(ROI_name)
plt.show()

# plot MACD:
plt.figure(figsize=(20,8))
plt.plot(np.arange(N)*TR, x, color='k', label='data')
plt.plot(np.arange(N)[N_EWMA1:]*TR, EWMAvec1, color='r', label='fast EWMA')
plt.plot(np.arange(N)[N_EWMA2:]*TR, EWMAvec2, color='g', label='slow EWMA')
plt.legend()
plt.xlabel('time')
plt.ylabel('BOLD response')
plt.xlim([0, N*TR*0.25]) #zoom inot first 25% of the chart
plt.title(ROI_name)
plt.show()

# plot RSI:
plt.figure(figsize=(20,8))
#plt.plot(np.arange(N)*TR, x, color='k', label='data')
plt.plot(np.arange(N)[N_RSI+1:]*TR, RSIvec, color='r', label='RSI')
plt.legend()
plt.xlabel('time')
plt.ylabel('BOLD response')
plt.xlim([0, N*TR*0.25]) #zoom inot first 25% of the chart
plt.title(ROI_name)
plt.show()

# plot Bollinger bands:
plt.figure(figsize=(20,8))
plt.plot(np.arange(N)*TR, x, color='k', label='data')
plt.plot(np.arange(N)[N_BOLL:]*TR, BOLLvec_ma, color='r', label='MA')
plt.plot(np.arange(N)[N_BOLL:]*TR, BOLLvec_upper, color='g', label='upper band')
plt.plot(np.arange(N)[N_BOLL:]*TR, BOLLvec_lower, color='b', label='lower band')
plt.legend()
plt.xlabel('time')
plt.ylabel('BOLD response')
plt.xlim([0, N*TR*0.25]) #zoom inot first 25% of the chart
plt.title(ROI_name)
plt.show()



# Correlation between different areas


plt.figure(figsize=(10,8))
plt.plot(data[:,1], data[:,2], color='k', label='data')  # np.arange(N)*TR
plt.legend()
plt.xlabel('ROI0')
plt.ylabel('ROI1')
#plt.xlim([0, N*TR*0.25]) #zoom inot first 25% of the chart
plt.title('Correlation between signals from different brain areas')
plt.show()



#
# Pouya's code (inspired from github.com/hmelo)
#



# In[Data Preperation]

seq_len = 22
nb_features = 1#len(df.columns)
data = df.loc[:,['V1', 'V2']].as_matrix() 
nb_features = data.shape[0]#1#len(df.columns)
sequence_length = seq_len + 1 # index starting from 0
result = []

for index in range(len(data) - sequence_length): # maxmimum date = lastest date - sequence length
    result.append(data[index: index + sequence_length]) # index : index + 22days

result = np.array(result)
row = round(0.9 * result.shape[0]) # 90% split

X_train = result[:int(row),:-1,:] # all data until day m
y_train = result[:int(row),-1,-1] # day m + 1 adjusted close price

X_test = result[int(row):,:-1,:]
y_test = result[int(row):,-1,-1] 

#X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], nb_features))
#X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], nb_features))  


# In[Model Building]

d = 0.2
shape = [nb_features, seq_len, 1] # feature, window, output
neurons = [128, 128, 32, 1]

model = []
model = Sequential()
model.add(LSTM(units=neurons[0], return_sequences=True, input_shape=(X_train.shape[1],X_train.shape[2]))) 
model.add(Dropout(d))
model.add(LSTM(neurons[1], return_sequences=False, input_shape=(X_train.shape[1],X_train.shape[2])))  
model.add(Dropout(d))
model.add(Dense(neurons[2],kernel_initializer="uniform",activation='relu'))        
model.add(Dense(neurons[3],kernel_initializer="uniform",activation='linear'))
#model.add(Activation('linear'))
model.compile(loss='mse', optimizer='rmsprop')
#model.compile(loss='mse',optimizer='adam', metrics=['accuracy'])
model.summary()


# In[Model Fitting]

model.fit(X_train, y_train, batch_size=512, epochs=10, validation_split=0.1, verbose=1)

# In[Results]

trainScore = model.evaluate(X_train, y_train, verbose=0)
print('Train Score: %.5f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))

testScore = model.evaluate(X_test, y_test, verbose=0)
print('Test Score: %.5f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))


from scipy import stats
from scipy.stats.stats import pearsonr

prediction = model.predict(X_test)
prediction = np.squeeze(prediction)
pearsonr(prediction,y_test)

#percentage_diff=[]
#for u in range(len(y_test)): # for each data index in test data
#    pr = prediction[u] # pr = prediction on day u
#    percentage_diff.append((pr-y_test[u]/pr)*100)
#    
#    
    
# In[Visualization]

plt.plot(prediction, color='red', label='Prediction')
plt.plot(y_test,color='blue', label='Actual')
plt.legend(loc='best')
plt.title('The test result for {}'.format('Resting_state Data'))
plt.xlabel('Volumes')
plt.ylabel('Voxel Value')
plt.show()




#############################################################################3


#
# Jean-Loup Loyer's code (from Udacity AIND time series prediction assignment)
#


# Load in and normalize the dataset

from my_answers import *
#dataset = np.loadtxt('normalized_apple_prices.csv')
dataset = df.loc[:,'V1'].as_matrix() 

# lets take a look at our time series
plt.plot(dataset)
plt.xlabel('time period')
plt.ylabel('Brain data')

# Implement the function window_transform_series in the file my_answers.py
from my_answers import window_transform_series

# window the data using your windowing function
window_size = 2
X,y = window_transform_series(series = dataset, window_size = window_size)


# split our dataset into training / testing sets
train_test_split = int(np.ceil(2*len(y)/float(3)))   # set the split point

# partition the training set
X_train = X[:train_test_split,:]
y_train = y[:train_test_split]

# keep the last chunk for testing
X_test = X[train_test_split:,:]
y_test = y[train_test_split:]

# NOTE: to use keras's RNN LSTM module our input must be reshaped to [samples, window size, stepsize] 
X_train = np.asarray(np.reshape(X_train, (X_train.shape[0], window_size, 1)))
X_test = np.asarray(np.reshape(X_test, (X_test.shape[0], window_size, 1)))




### Create required RNN model
# import keras network libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras

# given - fix random seed - so we can all reproduce the same results on our default time series
np.random.seed(0)


# TODO: implement build_part1_RNN in my_answers.py
from my_answers import build_part1_RNN
model = build_part1_RNN(window_size)

# build model using keras documentation recommended optimizer initialization
optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# compile the model
model.compile(loss='mean_squared_error', optimizer=optimizer)


# run your model!
model.fit(X_train, y_train, epochs=100, batch_size=50, verbose=1)


# generate predictions for training
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)


# print out training and testing errors
training_error = model.evaluate(X_train, y_train, verbose=0)
print('training error = ' + str(training_error))

testing_error = model.evaluate(X_test, y_test, verbose=0)
print('testing error = ' + str(testing_error))




### Plot everything - the original series as well as predictions on training and testing sets
import matplotlib.pyplot as plt
%matplotlib inline

# plot original series
plt.plot(dataset, color = 'k')

# plot training set prediction
split_pt = train_test_split + window_size 
plt.plot(np.arange(window_size,split_pt,1),train_predict,color = 'b')

# plot testing set prediction
plt.plot(np.arange(split_pt,split_pt + len(test_predict),1),test_predict,color = 'r')

# pretty up graph
plt.xlabel('day')
plt.ylabel('Brain data')
plt.legend(['original series','training fit','testing fit'],loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


### Correlation between original and predicted values, from training and test sets



plt.plot(train_predict, dataset[:len(train_predict)], color = 'k', label='Actual')
#plt.plot(train_predict, color='red', label='Prediction')
plt.legend(loc='best')
plt.title('The test result for {}'.format('Resting_state Data'))
plt.xlabel('Volumes')
plt.ylabel('Voxel Value')
plt.show()


plt.plot(test_predict, dataset[-len(test_predict):], color = 'k', label='Actual')
#plt.plot(train_predict, color='red', label='Prediction')
plt.legend(loc='best')
plt.title('The test result for {}'.format('Resting_state Data'))
plt.xlabel('Volumes')
plt.ylabel('Voxel Value')
plt.show()
