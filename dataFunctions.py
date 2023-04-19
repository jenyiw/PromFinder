# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 22:40:39 2023

@author: Jenyi

Code for managing data for learning models
"""

import os
import re
import pandas as pd
import numpy as np


def split_data(features_arr,
			   phylo_arr,
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
		train_data: features of training data
		train_label: labels of training data
		val_data: features of validation data
		val_label: labels of validation data
    """
    indices_arr = np.arange(features_arr.shape[0])
	
	

    if shuffle:
        np.random.shuffle(indices_arr)
        features_arr = features_arr[np.argsort(indices_arr)]
        phylo_arr = phylo_arr[np.argsort(indices_arr)]		
        label_arr = label_arr[np.argsort(indices_arr)]
		
    if train_proportion != 1:
	    split_index = int(len(features_arr)*train_proportion)
    else:
        split_index = int(len(features_arr))
		
    train_data = features_arr[:split_index]
    train_label = label_arr[:split_index]
    train_phylo = phylo_arr[:split_index]
	
	
    if train_proportion != 1:
        val_data = features_arr[split_index:]	
        val_label = label_arr[split_index:]
		
    else:
        val_data = None		
        val_label = None
	
    return train_data, train_label, val_data, val_label, train_phylo
	
def read_data(folder, file_suffix):
	
    """
	Load data
	
	Parameters:
		folder: str
			Folder where data is stored
	Returns:
		data_arr: numpy array

    """		
	
    file_names = os.listdir(folder)
    label_list = []
    for file in file_names:
        if not re.match(f'chr\d+_{file_suffix}', file):
            continue
        label_arr = np.load(os.path.join(folder, file))
        label_list.append(label_arr)
			
    label_arr = np.concatenate(label_list)	

    return label_arr	

def checkdir(folder_path, classifier):

    #create output folders if does not exist
    output_path = os.path.join(folder_path, 'output', f'{classifier}')
    if not os.path.exists(output_path):
            os.mkdir(output_path)
		
    #check if genome data exists
    genome_path = os.path.join(folder_path, 'genome_data')
    if not os.path.exists(genome_path):
            os.mkdir(genome_path)
            print('Please download genome data.')
            return
		
    #check if genome data exists
    cage_path = os.path.join(folder_path, 'cage_data')
    if not os.path.exists(cage_path):
            os.mkdir(cage_path)
            print('Please download CAGE data.')
            return		
		
    #check if kmer folder exists
    kmer_folder = os.path.join(folder_path, 'kmer_data')
    if not os.path.exists(kmer_folder):
            os.mkdir(kmer_folder)	

    #check if phyloP path exists
    phylop_path = os.path.join(folder_path, 'phyloP_data')
    if not os.path.exists(phylop_path):
            os.mkdir(phylop_path)
            print('Please download PhyloP data.')

	 

