# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 22:40:39 2023

@author: Jenyi

Code for managing data for learning models
"""

import os
import pandas as pd
import numpy as np


def split_data(features_arr,
			   label_arr,
			   train_proportion:float=0.5,
			   shuffle:bool=True):
	
    """
	Split the data into training and testing datasets
	Parameters:
		features_arr: numpy array
			 Matrix of (# samples, # features) of features for each sample
	   label_arr: numpy array
		   Matrix of (# samples,) of predicted labels
		train_proportion: float
			Proportion of whole dataset to use for training
		shuffle: bool
			Whether or not to shuffle the dataset
	Returns:
		train_data[:, :-1]: features of training data
		train_data[:, -1]: labels of training data
		test_data[:, :-1]: features of test data
		test_data[:, -1]: labels of test data
    """

    whole_data = np.concatenate((features_arr, label_arr), axis=1)
    if shuffle:
        np.random.shuffle(whole_data)
    split_index = int(len(whole_data)*train_proportion)
    train_data = whole_data[:split_index]
    test_data = whole_data[split_index:]
	
    return train_data[:, :-1], train_data[:, -1], test_data[:, :-1], test_data[:, -1]
	
	
