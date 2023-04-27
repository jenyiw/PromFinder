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
			Proportion of whole dataset to use for training and not validation
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
	
def read_data(folder, chromosome_list, file_suffix):
	
    """
	Load data
	
	Parameters:
		folder: str
			Folder where data is stored
	Returns:
		data_arr: numpy array

    """		
	
    file_names = os.listdir(folder)
    data_list = []
    for file in file_names:
        if not re.match(f'chr\d+_{file_suffix}', file):
            continue
        current_chromosome = re.search('chr\d+', file).group(0)
        if current_chromosome not in chromosome_list:
            continue
        data_arr = np.load(os.path.join(folder, file))
        data_list.append(data_arr)
			
    data_arr = np.concatenate(data_list)	

    return data_arr	

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
			
def check_specific_files(chromosome, kmer_path):

    folder_path = os.path.dirname(kmer_path)
	
    pause = False
	
    #check phylo P path
    genome_path = os.path.join(folder_path, 'genome_data')
    files = os.listdir(genome_path)

    if f'{chromosome}.fa' not in files:
        pause = True  
		
    return pause

def get_chrom_list(kmer_folder):
	
	files = os.listdir(kmer_folder)
	chr_list = []
	for f in files:
		if re.match('chr\d+_features_all.npy', f):
			current_chromosome = re.search('chr\d+', f).group(0)
			chr_list.append(current_chromosome)			
# 			if re.match('chr\d+_phylop.npy', f):
# 				chr_list.append(current_chromosome)
				
	return chr_list
	
				

	 

