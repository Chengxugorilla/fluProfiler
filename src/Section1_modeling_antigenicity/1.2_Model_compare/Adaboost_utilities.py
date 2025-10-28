#!/usr/bin/env python
# coding: utf-8

# # Model utilities
# It includes self defined functions for used models

# ### Imports

# In[ ]:


import numpy as np
from time import time
import random

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.preprocessing import MinMaxScaler

# for reproduciblility, fix the randomly generated numbers
SEED = 100


# ## Baseline model
# AdaBoost with default hyper-parameters
# 
# > **Parameters**
# > - X_train (numpy array): input features to train the model
# > - y_train (numpy array): output labels for supervised learning of the model
# > - X_test (numpy array): input features to test the model
# > - y_test: dummy, not used, default=None
# 
# > **Returns**
# > - results (dict): dictionary including:
# >    - pred_train (numpy array): predictions for training dataset
# >    - pred_test (numpy array): predictions for test dataset
# >    - model (object): trained model

# In[ ]:


def model_baseline(X_train, y_train, X_test, y_test=None):
    
    '''
    Model
    '''
    model = AdaBoostRegressor(random_state = SEED)
    
    '''
    Training
    '''
    time_start = time()
    model.fit(X_train, y_train)
    time_end = time()
    print(f"Time for training: {time_end - time_start}")
    
    '''
    Testing
    '''
    results = {}
    results['pred_train'] = model.predict(X_train)
    results['pred_test']  = model.predict(X_test)
    results['model']      = model
    
    return results


# ## Optimized AdaBoost
# AdaBoost regressor with optimized hyper-parameters for its top mutation matrix GIAG010101
# 
# > **Parameters**
# > - X_train (numpy array): input features to train the model
# > - y_train (numpy array): output labels for supervised learning of the model
# > - X_test (numpy array): input features to test the model
# > - y_test: dummy, not used, default=None
# 
# > **Returns**
# > - results (dict): dictionary including:
# >    - pred_train (numpy array): predictions for training dataset
# >    - pred_test (numpy array): predictions for test dataset
# >    - model (object): trained model

# In[ ]:


def model_AdaBoost(X_train, y_train):
    
    '''
    Model
    '''
    model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=1860, max_features=0.393686389369039),
                              n_estimators=230,
                              learning_rate=1.39248292746222,
                              random_state=SEED)
    
    '''
    Training
    '''
    time_start = time()
    model.fit(X_train, y_train)
    time_end = time()
    print(f"Time for training: {time_end - time_start}")
    
    '''
    Testing
    '''
    results = {}
    results['pred_train'] = model.predict(X_train)
    results['model']      = model
    
    return results


# ## Optimized AdaBoost for binary
# AdaBoost regressor with optimized hyper-parameters for binary encoding. This is used for NextFlu matched parameters simulation.
# 
# > **Parameters**
# > - X_train (numpy array): input features to train the model
# > - y_train (numpy array): output labels for supervised learning of the model
# > - X_test (numpy array): input features to test the model
# > - y_test: dummy, not used, default=None
# 
# > **Returns**
# > - results (dict): dictionary including:
# >    - pred_train (numpy array): predictions for training dataset
# >    - pred_test (numpy array): predictions for test dataset
# >    - model (object): trained model

# In[ ]:


def model_AdaBoost_binary(X_train, y_train, X_test, y_test=None):
    
    '''
    Model
    '''
    model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=7040, max_features=0.419171992638116),
                              n_estimators=410,
                              learning_rate=1.26852534318595,
                              random_state=SEED)
    
    '''
    Training
    '''
    time_start = time()
    model.fit(X_train, y_train)
    time_end = time()
    print(f"Time for training: {time_end - time_start}")
    
    '''
    Testing
    '''
    results = {}
    results['pred_train'] = model.predict(X_train)
    results['pred_test']  = model.predict(X_test)
    results['model']      = model
    
    return results


# ## Optimized RF model
# RF model with optimized hyper-parameters for its top mutation matrix AZAE970101
# 
# > **Parameters**
# > - X_train (numpy array): input features to train the model
# > - y_train (numpy array): output labels for supervised learning of the model
# > - X_test (numpy array): input features to test the model
# > - y_test: dummy, not used, default=None
# 
# > **Returns**
# > - results (dict): dictionary including:
# >    - pred_train (numpy array): predictions for training dataset
# >    - pred_test (numpy array): predictions for test dataset
# >    - model (object): trained model

# In[ ]:


def model_RF(X_train, y_train, X_test, y_test=None):
    
    '''
    Model
    '''
    model = RandomForestRegressor(n_estimators = 125,
                                  min_samples_split = 10,
                                  min_samples_leaf = 1,
                                  max_features = 0.375553860442328,
                                  max_depth = 200,
                                  bootstrap = True,
                                  random_state = SEED,
                                  n_jobs = -1)
    
    '''
    Training
    '''
    time_start = time()
    model.fit(X_train, y_train)
    time_end = time()
    print(f"Time for training: {time_end - time_start}")
    
    '''
    Testing
    '''
    results = {}
    results['pred_train'] = model.predict(X_train)
    results['pred_test']  = model.predict(X_test)
    results['model']      = model
    
    return results


# ## eXtreme Gradient Boosting (XGBoost)
# XGBoost regressor with optimized hyper-parameters for its top mutation matrix GIAG010101
# 
# > **Parameters**
# > - X_train (numpy array): input features to train the model
# > - y_train (numpy array): output labels for supervised learning of the model
# > - X_test (numpy array): input features to test the model
# > - y_test: dummy, not used, default=None
# 
# > **Returns**
# > - results (dict): dictionary including:
# >    - pred_train (numpy array): predictions for training dataset
# >    - pred_test (numpy array): predictions for test dataset
# >    - model (object): trained model

# In[ ]:


def model_XGBoost(X_train, y_train, X_test, y_test=None):
    
    '''
    Model
    '''
    model = XGBRegressor(booster='gbtree',
                         n_estimators=343, max_depth=23,
                         learning_rate=0.0586498853490469, subsample=0.790391730792872,
                         colsample_bytree=0.829414276718852, colsample_bylevel=0.360570017142831,
                         n_jobs=-1, random_state = SEED)
    
    
    '''
    Training
    '''
    time_start = time()
    model.fit(X_train, y_train,
              verbose=False)
    time_end = time()
    print(f"Time for training: {time_end - time_start}")
    
    '''
    Testing
    '''
    results = {}
    results['pred_train'] = model.predict(X_train)
    results['model']      = model
    results['pred_test']  = model.predict(X_test)
    
    return results


# ## Multi-layer Perceptron
# Multi-layer Perceptron with optimized hyperparameters for mutation matrix WEIL970102
# 
# > **Parameters**
# > - X_train (numpy array): input features to train the model
# > - y_train (numpy array): output labels for supervised learning of the model
# > - X_test (numpy array): input features to test the model
# > - y_test: dummy, not used, default=None
# 
# > **Returns**
# > - results (dict): dictionary including:
# >    - pred_train (numpy array): predictions for training dataset
# >    - pred_test (numpy array): predictions for test dataset
# >    - model (object): trained model

# In[ ]:
# ## ResNet
# Residual neural network with optimized hyperparameters for mutation matrix MUET010101
# 
# > **Parameters**
# > - X_train (numpy array): input features to train the model
# > - y_train (numpy array): output labels for supervised learning of the model
# > - X_test (numpy array): input features to test the model
# > - y_test: dummy, not used, default=None
# 
# > **Returns**
# > - results (dict): dictionary including:
# >    - pred_train (numpy array): predictions for training dataset
# >    - pred_test (numpy array): predictions for test dataset
# >    - model (object): trained model

