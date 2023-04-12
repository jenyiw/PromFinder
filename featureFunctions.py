# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 17:02:41 2023

@author: Jenyi

Gets features for each kmer
"""
import os
import pandas as pd
import numpy as np
import kmerFunctions as kF


def get_features(kmer_folder):
	
    """
	Get features for each kmer
	
	Parameters:
		kmer_folder: str
			Folder where the kmers are stored
	Returns:
		None
    """
	
    file_names = os.listdir(kmer_folder)
    gc_list = []
    for file in file_names:
        if '.npy' in file:
            continue

        kmer_df = pd.read_csv(os.path.join(kmer_folder, file), header=None, index_col=None)
        gc_list.append(calculate_gc(kmer_df))
			
    gc_arr = np.concatenate(gc_list)	

    return gc_arr	
		
def calculate_gc(kmer_df):
	
    """
	Calculate GC content of each kmer
	
	Parameters:
		kmer_df: DataFrame
			DataFrame containing sequences of all kmers
	Returns:
		gc_arr: numpy array
			Matrix of (# samples, ) of the GC content of each sample
    """	
    count = 0
    gc_arr = np.zeros(len(kmer_df))
    for i in range(len(kmer_df)):
        kmer = kmer_df.iloc[i, 0]
        for n in kmer:
            if n in ['c', 'g']:
                count += 1
	
        gc_arr[i] = count/len(kmer)
	
    return gc_arr

