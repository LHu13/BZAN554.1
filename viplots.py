import numpy as np
import pandas as pd
import tensorflow as tf
import os
import datetime
import random
import tensorflow_addons as tfa
import keras
import copy

#set wd
os.chdir('C:/Users/Jackson DeBord/Documents/Spring_23/Learning/Project1')

#load in model

#respecify architecture
model = keras.Sequential()
model.add(keras.layers.Dense(units=36, name='input'))
model.add(keras.layers.Dense(units=22, activation="sigmoid", name = 'hidden1')) #reduce hidden layer by 40%
model.add(keras.layers.Dense(units=14, activation="sigmoid", name= 'hidden2')) #reduce hidden layer by 40%
model.add(keras.layers.Dense(units=9, activation="sigmoid", name= 'hidden3')) #reduce hidden layer by 40%
model.add(keras.layers.Dense(units=1, activation="linear", name= 'output')) #outputs 1 result


#load in model
#opt = keras.optimizers.Adam(learning_rate=0.01)
#model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_squared_error'])

model.built = True
model.load_weights('trained_weights_0.h5')


#need to get dummies and divide by range

#read in pricing
pricing = pd.read_csv('pricing.csv')

#get ranges/min

ranges = pd.DataFrame({'range_quantity': [np.max(pricing['quantity']) - np.min(pricing['quantity'])],
                       'range_duration': [np.max(pricing['duration']) - np.min(pricing['duration'])],
                       'range_order': [np.max(pricing['order']) - np.min(pricing['order'])],
                       'range_price': [np.max(pricing['price']) - np.min(pricing['price'])],
                       'min_quantity': [np.min(pricing['quantity'])],
                       'min_duration': [np.min(pricing['duration'])],
                       'min_order': [np.min(pricing['order'])],
                       'min_price': [np.min(pricing['price'])]})


df_category = pd.read_csv('category_levels.csv').sort_values(by=["category_levels"])

#sets categories as 1 dimensional series
sorted_columns = df_category["category_levels"].values.tolist()

#start with order

#read in test pricing data
pricing = pd.read_csv('pricing_test.csv', names = ['sku','price','quantity',
                                                   'order', 'duration', 'category'])


#drop sku
pricing = pricing.drop(['sku'], axis = 1)

#scaling
pricing['quantity'] = (pricing['quantity'] - ranges['min_quantity'][0]) / ranges['range_quantity'][0]
pricing['duration'] = (pricing['duration'] - ranges['min_duration'][0]) / ranges['range_duration'][0]
pricing['order'] = (pricing['order'] - ranges['min_order'][0]) / ranges['range_order'][0]
pricing['price'] = (pricing['price'] - ranges['min_price'][0]) / ranges['range_price'][0]

#saving actual y-values
y_true = pricing['quantity'].to_numpy()

#dropping quantity
pricing = pricing.drop(['quantity'],axis = 1)



#changing to categorical data types
category_types = pd.CategoricalDtype(categories = sorted(df_category['category_levels'][:]), ordered = True)
#create dummies
pricing['category'] = pricing['category'].astype(category_types)


pricing = pd.get_dummies(pricing, ['category'])


#save unaltered data to numpy array
X_normal = pricing.to_numpy()

#permute variable of interest
pricing['order'] = np.random.permutation(pricing['order'])

#save as numpy array
X_permute = pricing.to_numpy()


#predict for reg data
y_hat_normal = model.predict(X_normal)

#predict for permuted data
y_hat_permute = model.predict(X_permute)

#calculate correlation for each
r_normal = np.corrcoef(y_true, y_hat_normal.flatten())
r_permute = np.corrcoef(y_true, y_hat_permute.flatten())

#find difference (vi)
vi_order = r_normal[0,1] - r_permute[0,1]
vi_order


#duration
pricing = pd.read_csv('pricing_test.csv', names = ['sku','price','quantity',
                                                   'order', 'duration', 'category'])


#drop sku
pricing = pricing.drop(['sku'], axis = 1)

#scale
pricing['quantity'] = (pricing['quantity'] - ranges['min_quantity'][0]) / ranges['range_quantity'][0]
pricing['duration'] = (pricing['duration'] - ranges['min_duration'][0]) / ranges['range_duration'][0]
pricing['order'] = (pricing['order'] - ranges['min_order'][0]) / ranges['range_order'][0]
pricing['price'] = (pricing['price'] - ranges['min_price'][0]) / ranges['range_price'][0]

#save y values and drop from df
y_true = pricing['quantity'].to_numpy()

pricing = pricing.drop(['quantity'],axis = 1)



#create dummies
category_types = pd.CategoricalDtype(categories = sorted(df_category['category_levels'][:]), ordered = True)
pricing['category'] = pricing['category'].astype(category_types)


pricing = pd.get_dummies(pricing, ['category'])

#save unaltered data
X_normal = pricing.to_numpy()

#permute duration
pricing['duration'] = np.random.permutation(pricing['duration'])

X_permute = pricing.to_numpy()

#predict from permuted data

y_hat_normal = model.predict(X_normal)

y_hat_permute = model.predict(X_permute)

r_normal = np.corrcoef(y_true, y_hat_normal.flatten())
r_permute = np.corrcoef(y_true, y_hat_permute.flatten())

#calculate vi
vi_duration = r_normal[0,1] - r_permute[0,1]
vi_duration



#price
pricing = pd.read_csv('pricing_test.csv', names = ['sku','price','quantity',
                                                   'order', 'duration', 'category'])



pricing = pricing.drop(['sku'], axis = 1)

#scale
pricing['quantity'] = (pricing['quantity'] - ranges['min_quantity'][0]) / ranges['range_quantity'][0]
pricing['duration'] = (pricing['duration'] - ranges['min_duration'][0]) / ranges['range_duration'][0]
pricing['order'] = (pricing['order'] - ranges['min_order'][0]) / ranges['range_order'][0]
pricing['price'] = (pricing['price'] - ranges['min_price'][0]) / ranges['range_price'][0]


y_true = pricing['quantity'].to_numpy()

pricing = pricing.drop(['quantity'],axis = 1)




category_types = pd.CategoricalDtype(categories = sorted(df_category['category_levels'][:]), ordered = True)
pricing['category'] = pricing['category'].astype(category_types)


pricing = pd.get_dummies(pricing, ['category'])

X_normal = pricing.to_numpy()


pricing['price'] = np.random.permutation(pricing['price'])

X_permute = pricing.to_numpy()


#fit predictions
y_hat_normal = model.predict(X_normal)

y_hat_permute = model.predict(X_permute)

r_normal = np.corrcoef(y_true, y_hat_normal.flatten())
r_permute = np.corrcoef(y_true, y_hat_permute.flatten())

#calculate vi
vi_price = r_normal[0,1] - r_permute[0,1]
vi_price

#category
pricing = pd.read_csv('pricing_test.csv', names = ['sku','price','quantity',
                                                   'order', 'duration', 'category'])



pricing = pricing.drop(['sku'], axis = 1)


pricing['quantity'] = (pricing['quantity'] - ranges['min_quantity'][0]) / ranges['range_quantity'][0]
pricing['duration'] = (pricing['duration'] - ranges['min_duration'][0]) / ranges['range_duration'][0]
pricing['order'] = (pricing['order'] - ranges['min_order'][0]) / ranges['range_order'][0]
pricing['price'] = (pricing['price'] - ranges['min_price'][0]) / ranges['range_price'][0]


y_true = pricing['quantity'].to_numpy()

pricing = pricing.drop(['quantity'],axis = 1)




category_types = pd.CategoricalDtype(categories = sorted(df_category['category_levels'][:]), ordered = True)
pricing_normal = copy.copy(pricing)

pricing_normal['category'] = pricing_normal['category'].astype(category_types)


pricing_normal = pd.get_dummies(pricing_normal, ['category'])

X_normal = pricing_normal.to_numpy()

pricing_permute = copy.copy(pricing)

pricing_permute['category'] = np.random.permutation(pricing_permute['category'])
pricing_permute['category'] = pricing_permute['category'].astype(category_types)


pricing_permute = pd.get_dummies(pricing_permute, ['category'])

X_permute = pricing_permute.to_numpy()



y_hat_normal = model.predict(X_normal)

y_hat_permute = model.predict(X_permute)

r_normal = np.corrcoef(y_true, y_hat_normal.flatten())
r_permute = np.corrcoef(y_true, y_hat_permute.flatten())


vi_category = r_normal[0,1] - r_permute[0,1]
vi_category

vi = [vi_duration, vi_price, vi_category, vi_order]

vi

import matplotlib.pyplot as plt

plt.bar(x = ['duration', 'price', 'category', 'order'],
        height = vi)

plt.title('Variable Importance Measures for Each Variable')
plt.xlabel('Variable')
plt.ylabel('Importance')
plt.show()
