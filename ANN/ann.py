# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 13:44:22 2020

@author: LENOVO
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]

geography=pd.get_dummies(X["Geography"],drop_first=True)
gender=pd.get_dummies(X["Gender"],drop_first=True)

X= pd.concat([X,geography,gender],axis=1)
X=X.drop(["Geography","Gender"],axis=1)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense
#from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout

classifier= Sequential()

classifier.add(Dense(output_dim=6, init = "he_uniform",activation="relu",input_dim=11))

classifier.add(Dense(output_dim = 6, init = 'he_uniform',activation='relu'))

classifier.add(Dense(output_dim = 1, init = 'glorot_uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'Adamax', loss = 'binary_crossentropy', metrics = ['accuracy'])

model=classifier.fit(X_train, y_train,validation_split=0.33, batch_size = 10, nb_epoch = 20)
classifier.evaluate(X_test, np.array(y_test))
 

y_pred= classifier.predict(X_test)
y_pred = (y_pred > 0.5)
 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred,y_test)





