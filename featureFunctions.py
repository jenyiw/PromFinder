# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 17:02:41 2023

@author: Jenyi

Gets features for each kmer
"""
import os
import re
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
    bend_seq = bend_df['Sequence'].to_list()
    bend_count = np.zeros((len(bend_seq)))
    bend_values = bend_df['lnp'].to_numpy()
    for file in file_names:
        if not re.match('chr\d+_kmer', file):
            continue
		
        print('Working on features for:', file)
        kmer_df = pd.read_csv(os.path.join(kmer_folder, file), header=None, index_col=None)
        feature_arr = np.zeros((len(kmer_df), 2))
		
        for i in range(len(kmer_df)):
            kmer = kmer_df.iloc[i, 0]
			
            gc_count = 0
            bd_sum = 0
            for n in range(len(kmer)):
				
				#calculate GC score
                if kmer[n] in ['c', 'g', 'C', 'G']:
                    gc_count += 1
					
                if (n != len(kmer)-1):
                    m = kmer[n:n+2].upper()	
                    if m in bend_seq:
                        bend_count[bend_seq.index(m)] += 1				
				
            feature_arr[i, 0] = gc_count/len(kmer)
            # feature_arr[i, 1] = calculate_tm(kmer)
            feature_arr[i, 1] = np.sum(bend_count*bend_values)	
			
            if i % 100 == 0:
                print(f'Processed {i} oligos')
			
        np.save(os.path.join(kmer_folder, 'chr21_features.npy'), feature_arr)

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
        if n not in count_dict.keys():
            continue
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
        bd_score = bend_df.loc[bend_df['Sequence'] == m, 'lnp'].values
        if len(bd_score) > 0:
            bd_sum += bd_score[0]
	
    return bd_sum

