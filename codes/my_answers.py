import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = [series[ii: ii + window_size] for ii in range(len(series) - window_size)]
    y = [series[ii + window_size] for ii in range(len(series) - window_size)]

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential([
            LSTM(units=5, input_shape=(window_size,1), activation='relu', return_sequences=False),
            Dense(1)
            ])
    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    text = ''.join(x for x in text if x in 'abcdefghijklmnopqrstuvwxyz' or x in punctuation or x == ' ')
    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = [text[ii: ii + window_size] for ii in range(0, len(text) - window_size, step_size)]
    outputs = [text[ii + window_size] for ii in range(0,len(text) - window_size, step_size)]

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape = (window_size, num_chars), return_sequences=False))
    model.add(Dense(num_chars, activation='linear'))
    model.add(Activation("softmax"))
    return model