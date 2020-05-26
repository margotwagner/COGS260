#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 22:30:28 2020

@author: samantha
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from numpy import genfromtxt
import math

# matplotlib inline

def load_data(filename):
    '''
    Load in data and return values
    
    param:   filename (str) - filename/path to data
    return:  data - data values
    '''
    # double check it's samples x features
    dataCSV = genfromtxt(filename, delimiter=',')
    data = pd.DataFrame(dataCSV)
    return data.values

def norm_data(data):
    '''
    Normalize data to z-values (0 mean and 1 std dev)
    
    param:   data - data values (n_samples, n_features)
    return:  data - normalized data
    '''
    means = data.mean(axis=0)    # mean for each feature
    stdevs = data.std(axis=0)    # std dev for each feature
    data = (data - means) / stdevs    # normalized data (Z-score)
    return data

from sklearn.decomposition import PCA
def pca(data):
    '''
    Does PCA on normalized data set. Can optionally set fewer components
    
    param: data - normalized data (n_samples, n_features)
    return: 
        data_pc: data transformed onto princinpal components
        components:  principal axes in feature space, array, shape (n_components, n_features)
        weights: percentage of variance explained by each of the selected components. array, shape (n_components,)
    '''
    # create PCA model
    pca = PCA() 
    
    # fit model to data
    data_pc = pca.fit(data)  
    
    # obtain components and components' weights
    components = pca.components_
    weights = pca.explained_variance_ratio_
    
    return data_pc, components, weights, pca

def cum_var_plot(weights, desired_var):
    '''
    Cumulative variance plot (number of components vs cumulative variance captured) with calculated number 
    of PCs required to get to a certain desired variance explained
    
    params:
        weights: percentage of variance explained by each of the selected components. array, shape (n_components,)
        desired_var:  percent variance to find number of PCs for
    return
        pcs_req:   pcs required to captured at least desired variance
        captured_var   exact variance captured by pcs_req
    
    '''
    INDEX_SHIFT = 1
    # cumulative variance captured
    cum_var = np.cumsum(weights) 
    
    # find pcs req to get desired variance
    pcs_req = math.ceil(np.min(np.where(cum_var > desired_var)))   
    
    # actual variance captured
    captured_var = cum_var[pcs_req-INDEX_SHIFT]
    
    # plotting
    plt.plot(range(INDEX_SHIFT,len(cum_var)+INDEX_SHIFT), cum_var)
    plt.axvline(x=pcs_req, ymin=0, ymax=1, color='k', linestyle='--')
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative variance captured')
    plt.title('Cumulative Variance Captured by Principal Components')
    
    return pcs_req, captured_var

def flip_data(data):
    # This is for flipping data in other direction (spatial)
    flipped_data = np.zeros((51,360))
    for i in range(51):
        flipped_data[i,:]=data[:,i]
    return flipped_data

import os
os.chdir('/Volumes/GoogleDrive/Shared drives/COGS 260 Project/Data/fragments3')
filename='s_l_60mA_processed_trial_1.csv'
d=load_data(filename)
norm_d=norm_data(d)
[data_pc, components, weights, pca]=pca(norm_d) # returns 51 components - this is time based, flipping returns 360 components
[pcs_req, captured_var]= cum_var_plot(weights, .9)


########  The code below can be used for saving the PCs if necessary #########
# Temporal PCs
os.chdir('/Volumes/GoogleDrive/Shared drives/COGS 260 Project/Data/fragments3')
for filename in os.listdir(os.getcwd()):
    os.chdir('/Volumes/GoogleDrive/Shared drives/COGS 260 Project/Data/fragments3')
    print(filename)

    d=load_data(filename)
    norm_d=norm_data(d)
    [data_pc, components, weights, pca]=pca(norm_d) # returns 51 components - this is time based
    [pcs_req, captured_var]= cum_var_plot(weights, .9)

    os.chdir('/Volumes/GoogleDrive/Shared drives/COGS 260 Project/Data/fragments_pca')
    os.chdir('/Volumes/GoogleDrive/Shared drives/COGS 260 Project/Data/fragments_pca')
    np.savetxt(filename[:26]+'_norm_d.csv', norm_d, delimiter=",")
    np.savetxt(filename[:26]+'_components.csv', components, delimiter=",")
    np.savetxt(filename[:26]+'_weights.csv', weights, delimiter=",")
    del d, data_pc, norm_d,components, weights, pca # This needs to be reset or there is an error "pca is not callable"

# Spatial PCs
# Will need to run above function first, skip Temporal PCs and run this.
for filename in os.listdir(os.getcwd()):
    os.chdir('/Volumes/GoogleDrive/Shared drives/COGS 260 Project/Data/fragments3')
    print(filename)

    d=load_data(filename)
    os.chdir('/Volumes/GoogleDrive/Shared drives/COGS 260 Project/Data/fragments3')
    # Need to clear vars before this
    d=load_data('s_l_60mA_processed_trial_1.csv')
    flipped_data=flip_data(d)
    norm_flipped=norm_data(flipped_data)
    [data_pc, components, weights, pca]=pca(norm_flipped) # returns 360 components - this is space based
    #[pcs_req_flipped, captured_var_flipped]= cum_var_plot(weights_flipped, .9)

