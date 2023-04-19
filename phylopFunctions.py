# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 16:21:22 2023

@author: Linda

Extract phyloP data
"""
import os
import re
import numpy as np


def get_phyloP_arr(phylo_path, 
				   positions,
				   kmer_folder,
				   half_window:int=300):
    phylop_arr = np.zeros((len(positions),))
    with open(os.path.join(phylo_path, 'chr21.phyloP46way.placental.wigFix'), 'r') as wig_file:
        current_chromosome = ''
        i = None
        j = 0
        for line in wig_file:
            # start_position = 0
            current_position = positions[j] - half_window
            if line.startswith('fixedStep'):
                if i == None:
                    current_chromosome = line.split('chrom=')[1].split(' ')[0]  # Extract chromosome name from header
                    start_position = int(line.split('start=')[1].split(' ')[0])
                    if start_position > current_position:
                        previous_start_position = start_position
                        continue
                    else:
                        start_position = previous_start_position
                        total_sum = 0
                        i = 0
					
            else:
                if i == 0:
                    score = line.split('\n')
                    score = float(score[0])
                    total_sum += score
                    i += 1
                    if i == 500:
                        j += 1
                        i = None
	
		#save positions
    np.save(os.path.join(kmer_folder, current_chromosome+'_phylop.npy'), phylop_arr)				
							
    return


		