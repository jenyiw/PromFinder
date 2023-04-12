# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 17:21:03 2023

@author: Jenyi

Main function for running the classifiers
"""
import os
import classifierFunctions as cF
import kmerFunctions as kF
import dataFunctions as dF
import featureFunctions as fF

os.chdir("..")
cage_path = r'./cage_data'
cage_file = 'refTSS_v3.0_human_coordinate.hg38.bed'
genome_path = r'./genome_data'

use_existing = True

def main(classifier):
	
    kmer_folder = r'./kmer_data'
	
    if not use_existing:
        kF.get_kmer_windows(genome_path, cage_path, cage_file)
	
	#get features and labels
    labels = kF.read_labels(kmer_folder)
    feature_arr = fF.get_features(kmer_folder)
	
	#check dimensions for column vectors
    if labels.ndim == 1:
        labels = labels.reshape(-1,1)
    if feature_arr.ndim == 1:
        feature_arr = feature_arr.reshape(-1, 1)
		
    print('Number of oligos:' , len(labels))
    print('Number of features:', feature_arr.shape[1])
	
	#split data
    train_data, train_label, test_data, test_label = dF.split_data(feature_arr, labels)
	
    print(f'Training {classifier}...')
    if classifier == 'svm':
        model = cF.create_svm(feature_arr, labels)
		
    elif classifier == 'deep_learning':
        model = cF.create_dl(feature_arr, labels)

    print(f'Testing {classifier}...')
    pred_label = cF.predict_svm(test_data, model)	

    metrics_list = cF.metrics_svm(test_label, pred_label)
	
    print(metrics_list)

    return metrics_list

if __name__ == "__main__":	
    os.chdir('.')
    metrics_list = main('svm')	
	
	
	