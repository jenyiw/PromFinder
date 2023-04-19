# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 16:21:22 2023

@author: Linda

Extract phyloP data
"""
import os
import re
import numpy as np
import json
def get_phyloP_dict(phylo_path):
	
    scores = {}

    file = r'chr21.phyloP46way.placental.wigfix'
    with open(os.path.join(phylo_path, file), 'r') as wig_file:
                for line in wig_file:
                    
                        if line.startswith('fixedStep'):
                                current_chromosome = line.split('chrom=')[1].split(' ')[0]  # Extract chromosome name from header
                                start_position = int(line.split('start=')[1].split(' ')[0])
                                scores[start_position] = []
					
                        else:
                            score = line.split('\n')
                            score = float(score[0])
                            scores[start_position].append(score)
                            # print(score)

		#save positions
    with open('chr21.phyloP46way.placental.json', 'w') as fp:
        json.dump(scores, fp)				



def get_phyloP_arr(phylo_path, 
				   positions,
				   kmer_folder,
				   half_window:int=250):
	
    print('-'*40)	
    print('Getting phylo P data!')

    phylop_arr = np.zeros((len(positions),))
    file_names = os.listdir(phylo_path)


    for file in file_names:
        current_chromosome = re.search('chr\d+', file).group(0)
        if re.match(f'chr\d+.phyloP46way.placental.json', file):
            f = open(os.path.join(phylo_path, file))
            scores = json.load(f)
	
    start_positions = list(scores.keys())
	
    s = 0
    last_checkpoint = 0
	
    for j in range(len(positions)):
		
        total_sum = 0		
        current_position = int(positions[j] - half_window)
		
        for s in range(last_checkpoint, len(start_positions)):
            temp_position = int(start_positions[s])
            if temp_position > current_position:
                continue
            else:
                last_checkpoint = s
                total_sum = 0
                missing_score = int(start_positions[s]) - current_position
                total_sum += np.sum(scores[start_positions[s-1]][-missing_score:])
                current_len = missing_score
                if current_len >= half_window*2:
                    break
                if s < len(start_positions)-1:
                    num_to_add = min(int(start_positions[s+1])-int(start_positions[s]), half_window*2-current_len)
                else:
                    total_sum += np.sum(scores[start_positions[s]])
                    break
                total_sum += np.sum(scores[start_positions[s]][:num_to_add])

        phylop_arr[j] = total_sum	
        if j % 1000 == 0:
            print(f'Processing {j} oligo')
				

	
    #save positions
    np.save(os.path.join(kmer_folder, current_chromosome+'_phylop.npy'), phylop_arr)				
							
    return

# get_phyloP_dict()
		