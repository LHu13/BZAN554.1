
############################## DATA PREPARATION ####################################
# Prepares the pricing_test.csv
#set wd to where pricing.csv is stored
import os
os.chdir('C:\\Users\\liana\\OneDrive\\Desktop')

######################################  PERFORMANCE TESTER  #########################################
# Tests the performance of the trained model on the test set

#imports the necessary packages
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

## MODEL CREATION
#creates the model architecture with sigmoid hidden layers and linear output
model = keras.Sequential()
model.add(keras.layers.Dense(units=36, name='input')) #36 inputs
model.add(keras.layers.Dense(units=22, activation="sigmoid", name = 'hidden1')) #reduce hidden layer by 40%
model.add(keras.layers.Dense(units=14, activation="sigmoid", name= 'hidden2')) #reduce hidden layer by 40%
model.add(keras.layers.Dense(units=9, activation="sigmoid", name= 'hidden3')) #reduce hidden layer by 40%
model.add(keras.layers.Dense(units=1, activation="linear", name= 'output')) #outputs 1 result

#compile and builds model
opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_squared_error'])
model.build((36,36))

#loads in the trained weights
model.load_weights("model_weights.h5")



## MODEL TESTING
#loads in the formatted pricing_test data
test_data = pd.read_csv("pricing_min_max_test.csv")

#creates the dataframe for the input values
dataframe_input = test_data[['price', 'order', 'duration', '0', '1', '2', '3', '4', '5', '6', '7',
                             '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                             '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29',
                             '30', '31', '32']]

#keeps only the values for the input
input_data = dataframe_input.values

#sets actual value of the test_data
y_true = test_data[['quantity']].values

#applies model onto the input data from the test dataset to predict output
y_pred = model.predict(input_data)

#calculates and prints R2
print(r2_score(y_true, y_pred))