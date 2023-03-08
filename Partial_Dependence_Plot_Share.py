#############################################################################################################
#imports the necessary packages
import os
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt 
from sklearn import linear_model
from keras.models import load_model
from keras.applications.vgg16 import VGG16
#from tensorflow.keras.models import load_model
from pycebox.ice import ice_plot
import h5py

os.chdir('C:\\Users\\strat\\Desktop\\MSBA\\554(DeepLearning)\\Project_1\\Final_Code')

# MODEL ARCHITECTURE
model = keras.Sequential()
model.add(keras.layers.Dense(units=36, name='input')) #36 inputs
model.add(keras.layers.Dense(units=22, activation="sigmoid", name = 'hidden1')) #reduce hidden layer by 40%
model.add(keras.layers.Dense(units=14, activation="sigmoid", name= 'hidden2')) #reduce hidden layer by 40%
model.add(keras.layers.Dense(units=9, activation="sigmoid", name= 'hidden3')) #reduce hidden layer by 40%
model.add(keras.layers.Dense(units=1, activation="linear", name= 'output')) #outputs 1 result

#Compiles and builds model
opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_squared_error'])
model.build((36,36))

#Loads in the trained weights
model.load_weights("trained_weights_0.h5")

all_data = pd.read_csv("pricing_min_max_test.csv")

test_data = all_data.iloc[:10000]

dataframe_input = test_data[['price', 'order', 'duration', '0', '1', '2', '3', '4', '5', '6', '7',
                             '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                             '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29',
                             '30', '31', '32']]

input_data = dataframe_input.values

y_true = test_data[['quantity']].values

y_pred = model.predict(input_data)

print(r2_score(y_true, y_pred))

ind = 1 #toggle

#Step 2.1
v = np.unique(input_data[:,ind]) #Returns sorted unique elements of an array

#Step 2.2
means = [] #Creates an empty list to fill
for i in v:
    #2.2.1: create novel data set where variable only takes on that value
    cte_X_cp = np.copy(input_data) #copies the cte_X array
    cte_X_cp[:,ind] = i
    #2.2.2 predict response
    yhat = model.predict(cte_X_cp)
    #2.2.3 mean
    means.append(np.mean(yhat)) #adds mean of predictions to end of list  

plt.plot(v, means)
plt.x.label('Scaled Order')
plt.y.label('Scaled Quantity')
plt.pyplot.title('Partial Dependence Plot - Order')
plt.show()
