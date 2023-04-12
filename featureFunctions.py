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
		feature_arr: numpy array
			Matrix of (# samples, # features) with features for each sample
    """
	
    file_names = os.listdir(kmer_folder)
    
	#get bendability scores
    bend_df = pd.read_csv(os.path.join(kmer_folder, 'bendability.csv'))
    for file in file_names:
        if '.npy' in file:
            continue

        kmer_df = pd.read_csv(os.path.join(kmer_folder, file), header=None, index_col=None)
        feature_arr = np.zeros((len(kmer_df), 3))
		
        for i in range(len(kmer_df)):
            kmer = kmer_df.iloc[i, 0]
            feature_arr[i, 0] = calculate_gc(kmer)
            feature_arr[i, 1] = calculate_tm(kmer)
            feature_arr[i, 2] = calculate_bend(kmer, bend_df)			

    return feature_arr	
		
def calculate_gc(kmer):
	
    """
	Calculate GC content of each kmer
	
	Parameters:
		kmer: str
			sequence of k-mers
	Returns:
		gc_score: float
			GC content
    """	
    count = 0
    for n in kmer:
        if n in ['c', 'g', 'C', 'G']:
                count += 1
	
    return count/len(kmer)

def calculate_tm(kmer):
	
    """
	Calculate melting temperature of k-mer
	
	Parameters:
		kmer: str
			sequence of k-mers
	Returns:
		tm_score: float
			Melting temperature
    """	
    count_dict = {'a':0, 'c':0, 't':0, 'g':0}
    for n in kmer:
        n = n.lower()
        count_dict[n] += 1
		
    tm = 64.9 + 41* (count_dict['g']+count_dict['c']-16.4)/len(kmer)
	
    return tm

def calculate_bend(kmer, bend_df):
	
    """
	Calculate bendability of k-mer
	
	Parameters:
		kmer: str
			sequence of k-mers
		bend_df: DataFrame
			DataFrame containing dinucleotide bendability from Bruckner et al., 2012
	Returns:
		bend_score: float
			Bendability
    """	
    bd_sum = 0
    for n in range(len(kmer)-1):
        m = kmer[n:n+2].upper()
        bd_score = bend_df.loc[bend_df['Sequence'] == m, 'lnp'].values[0]
        bd_sum += bd_score
	
    return bd_sum

