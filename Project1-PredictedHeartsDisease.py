# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 09:17:35 2022

@author: sabri
"""

import sklearn.datasets as skdatasets
from sklearn import preprocessing
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, metrics
#import tensorflow_transform as tft
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import datetime, os
import pandas as pd

#read data from csv
file_path= r"C:\Users\sabri\Documents\PYTHON\DL\Datasets\heart.csv"
heart_data= pd.read_csv(file_path, sep=',', header= 0 )
#%%
#View teh first 5 data
heart_data.head()

#%%
#Check null in all column
print(heart_data.isna().sum())

#DATA CLEANING IS DONE
#%%
#DATA PREPROCESSING
#5. Split into features and labels
feature= heart_data.copy()
label= heart_data.pop('target')

features_np= np.array(feature)
labels_np= np.array(label)

#%%
#train & test split

SEED=12345
x_train, x_test, y_train, y_test= train_test_split(feature, label, test_size=0.2, random_state=SEED)


#DAta normalisation
standardizer = StandardScaler()
standardizer.fit(x_train)
x_train= standardizer.transform(x_train)
x_test= standardizer.transform(x_test)

##DATA PREPARATION IS COMPLETED
#%%
#4. Construct NN model
nIn = x_train.shape[-1]
nClass = y_train.shape[-1]

#Use functional API
inputs = keras.Input(shape=(nIn,))
h1 = layers.Dense(128,activation='relu')
h2 = layers.Dense(64,activation='relu')
h3 = layers.Dense(32,activation='relu')
h4 = layers.Dense(16,activation='relu')
out = layers.Dense(1,activation='sigmoid')

x = h1(inputs)
x = h2(x)
x = h3(x)
x = h4(x)
outputs = out(x)

model = keras.Model(inputs=inputs,outputs=outputs)
model.summary()

#%%

#9. Compile and train the model
base_log_path = r"C:\Users\sabri\Documents\PYTHON\DL\TensorBoard\tb_logs"
log_path = os.path.join(base_log_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+ '___Project1')
es= EarlyStopping(monitor='val_loss',patience=10)
tb = TensorBoard(log_dir=log_path)

optimizers= optimizers.Adam(learning_rate= 0.0001)
loss= losses.BinaryCrossentropy(from_logits= False)
accuracy= metrics.BinaryAccuracy()

model.compile(optimizer=optimizers ,loss= loss ,metrics= accuracy)
history = model.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=32,epochs=1000, callbacks=[tb, es])






