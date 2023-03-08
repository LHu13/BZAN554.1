

#############################  MODEL TRAINING AND LOGGING ################################################
#trains model and logs the information

#loads in the necessary packages
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import keras
import psutil
from sklearn.metrics import r2_score

#set wd to where pricing.csv is stored
os.chdir('C:\\Users\\liana\\OneDrive\\Desktop')
#sets pricing data
datafile_name = "pricing.csv"

#### FOR LOGGING LIVE R2 #####
# The test data here is pre-processed to avoid extra computation on steps with performance reporting
# It is our opinion that this does not violate the row-by-row objective since the memory cost here is both known
# to be small and of fixed size
# NOTE THAT THIS DATA IS NOT USED IN ANY WAY TO IMPROVE THE PERFORMANCE OF THE MODEL, ONLY TO CHART THE
# PERFORMANCE AGAINST THIS DATA THROUGH TIME
test_all = pd.read_csv("pricing_min_max_test.csv")
test_inputs = test_all[['price', 'order', 'duration', '0', '1', '2', '3', '4', '5', '6', '7',
                               '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                               '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29',
                               '30', '31', '32']].values
test_output = test_all[['quantity']].values


## DATA STUFF
#read in ranges calculated from pricing.csv
df_ranges = pd.read_csv('ranges.csv')

#read in category dummy levels
df_category = pd.read_csv('category_levels.csv').sort_values(by=["category_levels"])

#sets categories as 1 dimensional series
sorted_columns = df_category["category_levels"].values.tolist()

#sets number of rows to read into memory and train on (chunk size of 1 according to last lecture)
CHUNK_SIZE = 1
#sets number of epochs
NUM_EPOCHS = 1

#worsens performance, but seems to stop memory growth in mb/epoch terms
class ObliterateMemoryLeak(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        #calls garbage collector
        gc.collect() 
        #clearning all the unnecessary non-weight data
        tf.keras.backend.clear_session()

## MODEL ARCHITECTURE
#creates the model architecture with sigmoid hidden layers and linear output
model = keras.Sequential()
model.add(keras.layers.Dense(units=36, name='input'))
model.add(keras.layers.Dense(units=22, activation="sigmoid", name = 'hidden1')) #reduce hidden layer by 40%
model.add(keras.layers.Dense(units=14, activation="sigmoid", name= 'hidden2')) #reduce hidden layer by 40%
model.add(keras.layers.Dense(units=9, activation="sigmoid", name= 'hidden3')) #reduce hidden layer by 40%
model.add(keras.layers.Dense(units=1, activation="linear", name= 'output')) #outputs 1 result

#Compile model
opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_squared_error'])


## LOGGING RAM AND R2 PERFORMANCE STUFF
#establishes base RAM
baseline_RAM_usage_GB = psutil.virtual_memory()[3]/1000000000.0
#sets when to start logging RAM 
RAM_LOGGING_INTERVAL_CHUNKS = 10
#sets when to start logging R2 performance
PERFORMANCE_LOGGING_INTERVAL_CHUNKS = RAM_LOGGING_INTERVAL_CHUNKS * 10
#sets the logging file name
LOGGING_FILENAME = "Logging.csv"

#ensures that there is no other logging file in the directory/removes old logging files
try:
    os.remove(LOGGING_FILENAME)
except OSError:
    pass

#sets logging file 
ram_file = open(LOGGING_FILENAME, "x")
#adds column title row to logging file
ram_file.write("step,ram,R^2,MSE\n")

#sets iterator to 1
step_iter = 1



## MODEL TRAINING

for epoch in range(0, NUM_EPOCHS): #runs for set number of epochs

    print("Starting new epoch! Epoch: " + str(epoch)) #confirms which epoch has started 

    for chunk in pd.read_csv(datafile_name, chunksize=CHUNK_SIZE): #loads in the pricing.csv data file in line by line

        # We can just check once for nulls in the single row case instead of dropping NA rows from a larger chunk
        if not chunk.isnull().values.any():

            #normalize input with computed min_max
            dataframe_input = (chunk[['price', 'order', 'duration']].values -
                               df_ranges[['min_price', 'min_order', 'min_duration']].values) / df_ranges[['range_price', 'range_order', 'range_duration']].values
            dataframe_input = pd.DataFrame(dataframe_input, columns=['price', 'order', 'duration'])

            #creates the dummies
            # get a column of [True, False] that indicates which dummy matches the category in the row just read,
            # this can be interpreted as [0, 1] by pandas
            dummies = pd.DataFrame([df_category["category_levels"].eq(chunk['category'].values[0]).values.tolist()], columns=sorted_columns)

            #need to match the indices so they concat correctly
            dataframe_input.index = dummies.index
            #combines them
            input_data = pd.concat([dataframe_input, dummies], axis=1).values
            #makes sure the input data is in float32
            X = np.float32(input_data)

            #perform min_max normalization on the output
            Y = (chunk[['quantity']].values - df_ranges[['min_quantity']].values) / df_ranges[['range_quantity']]
            
            #fits/trains model
            model.fit(X, Y, epochs=1, verbose=1, callbacks=ObliterateMemoryLeak())

            
            ## LOGGING RAM AND R2 PERFORMANCE 
            if step_iter % RAM_LOGGING_INTERVAL_CHUNKS == 0: #performs on every ram logging interval chunk
                #sets outstring string
                outstring = str()
                #adds the step it is and the RAM usage to string
                outstring = outstring + str(step_iter) + "," + str(psutil.virtual_memory()[3]/1000000000.0 - baseline_RAM_usage_GB) + ","

                #r2 and mse performance tracking
                if step_iter % PERFORMANCE_LOGGING_INTERVAL_CHUNKS == 0: #performs on every r2 logging interval chunk
                    y_pred = model.predict(test_inputs)
                    outstring = outstring + str(r2_score(test_output, y_pred)) + ","
                    test_loss = model.evaluate(X,Y)
                    outstring = outstring + str(test_loss[0])
                    #deletes to prep for new one
                    del y_pred
                ram_file.write(outstring + "\n") #writes the string as a new row into ram_file

                #make sure nothing is sitting around in memory waiting on disk IO
                #cleaner
                ram_file.flush()
                #deletes to prep for new one
                del outstring
                if step_iter % 1000 == 0:
                    #saves the model weights into a folder
                    model.save_weights("model/model_chkpnt_" + str(step_iter)+".h5")
                    print("Model saved")
            #deletes to prep for new one    
            del X, Y, input_data, dummies, dataframe_input
        #adds to iteration step
        step_iter += 1


    #checks that the epoch has ended
    print("Checkpointing epoch: " + str(epoch))





