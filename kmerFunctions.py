# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 14:44:15 2023

@author: Asus
"""
import os
import pandas as pd
import numpy as np


def get_kmer_windows(genome_path,
					 cage_path,
					 cage_file,
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

    cage_df = pd.read_csv(os.path.join(cage_path, cage_file), delimiter='\t',
					  header=None)
	
	#get predictions and save predictions	
    pred_path = r'./cage_data'
    pred_file = 'TSS.classification.hg38'
    pred_df = pd.read_csv(os.path.join(pred_path, pred_file), delimiter='\t')
	
	#get list of chromosomes
    chr_list_all = list(set(cage_df[0].tolist()))
    chr_list = [x for x in chr_list_all if len(x) < 6] #Note there is a weird chrM

    chr_list = ['chr21'] #temporary for trial

    #iterate over every chromosome 
    for ch in chr_list:
		
        print(f'Working on chromosome: {ch}')
        chr_cage_df = cage_df[cage_df[0] == ch]
        genome_file = os.path.join(genome_path, ch+'.fa')
		
		#data specifics for kmers
        kmer_list = []
        kmer_path = r'./kmer_data'
        kmer_file = ch+'_kmer.csv'
		
		#data specifics for kmers
        label_arr = np.zeros((len(chr_cage_df)))
        label_path = kmer_path
        label_file = ch+'_label.npy'
	
        #iterate over every possible tss
        with open(genome_file, 'r') as f:
            line = f.readlines()[1:]
            item_length = len(line[0])
            for k in range(len(chr_cage_df)):
				
				#get sequence of kmer
                kmer = get_kmer(chr_cage_df.iloc[k, 1], item_length, line)					
                kmer_list.append(kmer)
				
				#get labels
                label = get_label(chr_cage_df.iloc[k, 3], pred_df)
                label_arr[k] = label
				

		#save kmers
        kmer_df = pd.DataFrame({'sequence': kmer_list})             
        kmer_df.to_csv(os.path.join(kmer_path, kmer_file), header=None, index=None)
				
		#save labels
        np.save(os.path.join(label_path, label_file), label_arr)
				

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
				line):
	
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
	
    start_pos = center_pos - 250
    end_pos = center_pos + 250
			
    start_line = start_pos // item_length
    end_line = end_pos // item_length
    kmer = ''
    for j in range(start_line, end_line+1):
        if j == start_line:		
            kmer += line[j][start_pos % item_length:]
        elif j == end_line:
            kmer += line[j][:end_pos % item_length]	
        else:
            kmer += line[j]
			
    return kmer

def read_labels(kmer_folder):
	
    """
	Load labels of samples
	
	Parameters:
		kmer_folder: str
			Folder where kmer data is stored
	Returns:
		label_arr: numpy array
			Matrix of (# samples, ) of the label of each sample
    """		
	
    file_names = os.listdir(kmer_folder)
    label_list = []
    for file in file_names:
        if not '.npy' in file:
            continue
        label_arr = np.load(os.path.join(kmer_folder, file))
        label_list.append(label_arr)
			
    label_arr = np.concatenate(label_list)	

    return label_arr	

		

	
if __name__ == "__main__":	
    os.chdir('..')
    cage_path = r'./cage_data'
    cage_file = 'refTSS_v3.0_human_coordinate.hg38.bed'
    genome_path = r'./genome_data'
    get_kmer_windows(genome_path, cage_path, cage_file)

