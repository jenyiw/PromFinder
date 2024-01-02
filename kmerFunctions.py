# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 14:44:15 2023

@author: Jenyi
"""
import os
import re
import pandas as pd
import numpy as np
import dataFunctions as dF
import get_background as get_background

import time

def get_kmer_windows(genome_path,
					 cage_path,
					 cage_file,
					 kmer_path,
					 window:int=500,
					 train:bool=True,
					 ):
	
    """
	Get k-mer around the TSS identified by CAGE for classification
	
	Parameters:
		genome_path: str
			Folder where genome data is stored
		cage_path: str
			Folder where CAGE data is stored
		cage_file: str
			File name of CAGE data
			
	Returns:
		None
    """	

    if train:
        cage_df = pd.read_csv(os.path.join(cage_path, cage_file), delimiter='\t',
					  header=None)
        chr_col = 0
		
    else:
        cage_df = pd.read_csv(os.path.join(cage_path, cage_file), delimiter=',',
					  index_col=0)
        chr_col = 'chr'	

		
# 	#get predictions and save predictions	
#     pred_file = 'TSS.classification.hg38'
#     pred_df = pd.read_csv(os.path.join(cage_path, pred_file), delimiter='\t')
	
	#get list of chromosomes

    chr_list_all = list(set(cage_df[chr_col].tolist()))
    chr_list = [x for x in chr_list_all if len(x) < 6] #Note there is a weird chrM
	
    print(f'Found list of chromosomes: {chr_list}')

    #iterate over every chromosome 
    for ch in chr_list:
		
        print(f'Working on chromosome: {ch}')
		
		#check if all files exist for that chromosome, or else skip
        pause = dF.check_specific_files(ch, kmer_path)
		
        if pause:
            print(f'Missing file for {ch}...skipping')
            continue
		
        chr_cage_df = cage_df[cage_df[chr_col] == ch]
        genome_file = os.path.join(genome_path, ch+'.fa')
		
		#data specifics for kmers
        kmer_list = []
        positions_arr = []
        kmer_file = ch+'_kmer.csv'
        positions_file = ch+'_positions.npy'
		
		#data specifics for kmers
        label_arr = []
        label_path = kmer_path
        label_file = ch+'_label.npy'
	
        #iterate over every possible tss
        with open(genome_file, 'r') as f:
            line = f.readlines()[1:]
            item_length = len(line[0])
            for k in range(len(chr_cage_df)):
								
				#get sequence of kmer
                kmer = get_kmer(chr_cage_df.iloc[k, 1], item_length, line, half_window=window//2)
                if kmer.lower().count('n') > 0:
                    continue
                kmer_list.append(kmer)
                positions_arr.append(chr_cage_df.iloc[k, 1])
				
                label_arr.append(1)
				
                if k % 1000==0:
                    print(f'Processed {k} oligos...')
								
		
        positions_arr = np.array(positions_arr)
        #get labels
        label_arr = np.array(label_arr)

				
		#get background samples
        if train:
            print('Getting background samples')
            # drawn_positions = get_background.get_background(genome_file, os.path.join(cage_path, cage_file), ch, 1, window, len(positions_arr), list(positions_arr))						
		
#             neg_positions_arr = []
#             for k in range(len(drawn_positions)):

# 				#get sequence of negative kmer and add to previous list
#                 kmer = get_kmer(drawn_positions[k], item_length, line, half_window=window//2)
#                 if kmer.lower().count('n') > 0:
#                     continue
#                 kmer_list.append(kmer)	
#                 neg_positions_arr.append(drawn_positions[k])


        neg_positions_arr = []
        for k in range(len(label_arr)):
            kmer = ''.join(list(np.random.choice(['A', 'T', 'C', 'G'], size=(window))))
            kmer_list.append(kmer)
            neg_positions_arr.append(-1)

				
        neg_positions_arr = np.array(neg_positions_arr)
			
        negative_labels = np.zeros_like(neg_positions_arr)
			
        #concatenate to positive samples
        label_arr = np.concatenate((label_arr, negative_labels))
        positions_arr = np.concatenate((positions_arr, neg_positions_arr))
			

		#save kmers
        print(f'Total number of oligos: {len(kmer_list)}')
        kmer_df = pd.DataFrame({'sequence': ['']*len(kmer_list)})
        for i, k in enumerate(kmer_list):
            kmer_df.iloc[i, 0] = k             
        kmer_df.to_csv(os.path.join(kmer_path, kmer_file), header=None, index=None)
				
		#save labels
        np.save(os.path.join(label_path, label_file), label_arr)

		#save positions
        np.save(os.path.join(label_path, positions_file), positions_arr)
				

def get_label(refTSSID, 
				   pred_df):
	
    """
	Get ground truth labels
	
	Parameters:
		refTSSID: str
			refTSSID from CAGE data
		pred_df: DataFrame
			DataFrame from classified TSS data containing predictions
	Returns:
		label: int
			label of sample
    """
	
    label = pred_df.loc[pred_df['refTSSID'] == refTSSID, 'TSSclassification'].values[0]
    if label == 'yes':
        label = 1
    else:
        label = 0
	
    return label
		
def get_kmer(center_pos, 
				item_length,
				line,
				half_window:int=500):
	
    """
	Obtain kmer sequence for each CAGE position
	
	Parameters:
		center_pos: int
			Position of the possible transcription start site from CAGE data
		item_length:int
			Length of each line in the genome fasta file
		line: list
			list of lines from the genome fasta file
			
	Returns:
		kmer: str
			500 bp sequence around the CAGE proposed start site
    """	
	
    start_pos = center_pos - half_window
    end_pos = center_pos + half_window
			
    start_line = start_pos // item_length
    end_line = end_pos // item_length

	
    if end_line+1 > len(line):
        return 'n'
	
    kmer = ''
    for j in range(start_line, end_line+1):
        if j == start_line:		
            kmer += (line[j][start_pos % item_length:]).strip()
        elif j == end_line:
            kmer += (line[j][:end_pos % item_length]).strip()	
        else:
            kmer += (line[j]).strip()

	
    # kmer_list = [] 
    # for j in range(start_line, end_line+1):
    #     if j == start_line:		
    #         kmer_list.append((line[j][start_pos % item_length:]).strip())
        # elif j == end_line:
        #     kmer_list.append((line[j][:end_pos % item_length]).strip())
        # else:
        #     kmer_list.append((line[j]).strip())
			
    # kmer = ''.join(kmer_list)


    return kmer

		

	
if __name__ == "__main__":	
    os.chdir('..')
    cage_path = r'./cage_data'
    cage_file = 'refTSS_v3.0_human_coordinate.hg38.bed'
    genome_path = r'./genome_data'
    get_kmer_windows(genome_path, cage_path, cage_file)

