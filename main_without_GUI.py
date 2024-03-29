# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 17:21:03 2023

@author: Jenyi

Main function for running the classifiers
"""
import os
import pandas as pd
import pickle
import numpy as np
import classifierFunctions as cF
import kmerFunctions_v2 as kF
import dataFunctions as dF
import featureFunctions as fF
import phylopFunctions as pF
import cnnFunctions

import matplotlib.pyplot as plt

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 17:21:03 2023

@author: Jenyi

Main function for running the classifiers
"""

class PromFinder():
	
    def __init__(self, 
					 kmer_size:int=500,):

        self.kmer_size = kmer_size


    def create_kmer_data(self, 
						 folder_path,
						 cage_file,
						 train:bool=True):
		
        """
		Calculate features for every k-mer
		Parameters:
			folder_path: str
			   Path with the data folders
		   cage_file: str
		       name of CAGE file to use
	   
	   Returns:
		   feature_arr: (# of kmers, # of features, k) numpy array of features
		   phylo_arr: (# of kmers) numpy array of phylo scores
		   labels: (# of kmers,) numpy array of labels
		
		
        """
		  
        kmer_folder = os.path.join(folder_path, 'kmer_data')
        cage_path = os.path.join(folder_path, 'cage_data')
        genome_path = os.path.join(folder_path, 'genome_data')		
        phylop_path = os.path.join(folder_path, 'phyloP_data')
		
        chromosome_list = dF.get_chrom_list(genome_path)
		
        if not self.use_existing:
            kF.get_kmer_windows(genome_path, cage_path, cage_file, kmer_folder, window=self.kmer_size, train=train)
            fF.get_features(kmer_folder, window_size=self.kmer_size)
            # for ch in chromosome_list:
            #     positions_arr = dF.read_data(kmer_folder, [ch], 'positions')
            #     pF.get_phyloP_arr(phylop_path, positions_arr, ch, kmer_folder, half_window=self.kmer_size//2)		

	    #get features and labels
        feature_arr = dF.read_data(kmer_folder, chromosome_list, 'features_all')
        
        # phylo_arr = dF.read_data(kmer_folder, chromosome_list, 'phylop')

        labels = dF.read_data(kmer_folder, chromosome_list, 'label')
        phylo_arr = np.zeros_like(labels)		
	
        print('Number of oligos:' , len(labels))
        print('Number of features:', feature_arr.shape[-1])
		
        return feature_arr, phylo_arr, labels
	
    def train(self,
			  train_folder,
  			  cage_file_name,
			  classifier,
			  use_existing:bool=True):
	
        """
		Train model
		Parameters:
			train_folder: str
			   Path with the data folders
		   cage_file: str
		       name of CAGE file to use
		   classifier: str
		       classifier to use
		   use_existing: bool
				 Whether to use existing pre-calculated features   
	   
	   Returns:
		   None
		
		
        """		
		
        print('Calculating features...')
	
        self.use_existing = use_existing
		
        model_path = os.path.join(os.path.dirname(train_folder), 'models', f'{classifier}')
        if not os.path.exists(model_path):
            os.mkdir(model_path)
			
        feature_arr, phylo_arr, label_arr = self.create_kmer_data(train_folder, cage_file_name)
		
        print(f'Training {classifier}...')			
        if classifier == 'dl':
	   
            train_data, train_label, val_data, val_label, train_phylo = dF.split_data(feature_arr, phylo_arr, label_arr, train_proportion=0.9, shuffle=True)
			
            cnnFunctions.create_CNN(model_path,
					   train_data, train_label.reshape(-1,1),
					   val_data[0:1,...], val_label[0:1,...].reshape(-1,1))
			

        elif classifier == 'svm':
			
            train_data = np.max(feature_arr, axis=1)
# 	            phylo_arr = train_phylo.reshape(-1,1)
            # train_data = np.concatenate((train_data, phylo_arr.reshape(-1,1)), axis=1)
            save_model_path = os.path.join(model_path, 'svm_model.sav')
            cF.create_svm(train_data, label_arr.reshape(-1,), save_model_path)			
		
        elif classifier == 'dl_svm': 
           train_data, train_label, val_data, val_label, train_phylo = dF.split_data(feature_arr, phylo_arr, label_arr, train_proportion=0.9, shuffle=True)

           cnnFunctions.create_CNN(model_path,
					   train_data, train_label.reshape(-1,1),
					   val_data[0:1,...], val_label[0:1,...].reshape(-1,1))
			
           pred_label, x_out = cnnFunctions.predict_CNN(model_path, train_data, train_label.reshape(-1,1),
												  feed_svm=True)

            #concatenate data
           # sum_data = np.max(train_data, axis=1)
           # pred_label = np.concatenate((pred_label.reshape(-1,1), sum_data), axis=1)
          #  print(x_out.shape)
           save_model_path = os.path.join(model_path, 'svm_model.sav')
           cF.create_svm(x_out, train_label.reshape(-1,), save_model_path)		

			
        elif classifier == 'rf':

             train_data = np.max(feature_arr, axis=1)			
             # train_data = np.concatenate((train_data, phylo_arr.reshape(-1,1)), axis=1)
             save_model_path = os.path.join(model_path, 'rf_model.sav')				
             cF.create_rf(train_data, label_arr.reshape(-1,), save_model_path)

        elif classifier == 'dl_rf': 
           train_data, train_label, val_data, val_label, train_phylo = dF.split_data(feature_arr, phylo_arr, label_arr, train_proportion=0.9, shuffle=True)

          #  cnnFunctions.create_CNN(model_path,
					#    train_data, train_label.reshape(-1,1),
					#    val_data[0:1,...], val_label[0:1,...].reshape(-1,1))
			
           pred_label, x_out = cnnFunctions.predict_CNN(model_path, train_data, train_label.reshape(-1,1),
												  feed_svm=True)

            #concatenate data
          #  sum_data = np.max(train_data, axis=1)           			
          #  pred_label = np.concatenate((pred_label.reshape(-1,1), sum_data), axis=1)
           save_model_path = os.path.join(model_path, 'rf_model.sav')
           cF.create_rf(x_out, train_label.reshape(-1,), save_model_path)		
		   
    def predict(self, 
			  test_folder,
			  cage_file_name,			  
			  classifier,
			  model_path,			  
			  use_existing:bool=True):

        """
		Use pre-trained model for prediction
		Parameters:
			test_folder: str
			   Path with the data folders
		   cage_file: str
		       name of CAGE file to use
		   classifier: str
		       classifier to use
		   model_path: str
		       folder where models are stored		   
		   use_existing: bool
				 Whether to use existing pre-calculated features   
	   
	   Returns:
		   metrics_list : list
		        List containing calculated metrics
		   pred_labels: numpy array
		       Predicted labels for each sample
		
		
        """		

        print('Calculating features...')
		
        self.use_existing = use_existing
        output_path = os.path.join(test_folder, 'output')	
        test_data, test_phylo, test_label = self.create_kmer_data(test_folder, cage_file_name, train=False)	
		
        print(f'Testing {classifier}...')	
		
        if classifier == 'svm':
            test_data = np.max(test_data, axis=1)		
            pred_label = cF.predict(test_data, os.path.join(model_path, 'svm_model.sav'))
	
        elif classifier == 'rf':
            test_data = np.max(test_data, axis=1)			
            pred_label = cF.predict(test_data, os.path.join(model_path, 'rf_model.sav'))		
	
        elif classifier == 'dl':
		
            pred_label = cnnFunctions.predict_CNN(model_path, test_data, test_label.reshape(-1,1))

        elif classifier == 'dl_svm':
		
            prob, x_out = cnnFunctions.predict_CNN(model_path, test_data, test_label.reshape(-1,1))

			
			#run SVM
            save_model_path = os.path.join(model_path, 'svm_model.sav')
            pred_label = cF.predict(x_out, save_model_path)


        elif classifier == 'dl_rf':
		
            prob, x_out = cnnFunctions.predict_CNN(model_path, test_data, test_label.reshape(-1,1))
			
			#run rf
            save_model_path = os.path.join(model_path, 'rf_model.sav')
            pred_label = cF.predict(x_out, save_model_path)
			
        metrics_list = cF.metrics(test_label, pred_label)
	
        for m in metrics_list.keys():	
            print(f'{m}: {metrics_list[m]:.2f}')
			
        df = pd.DataFrame({'label': list(pred_label)})
		
        df.to_csv(os.path.join(output_path, 'final_pred.csv'))

        return metrics_list, pred_label
	

if __name__ == "__main__":
    os.chdir('..')	
    obj = PromFinder(kmer_size=1000)
    obj.train(r'./human',
 			  'refTSS_v3.0_human_coordinate.hg38.bed',
 			  'dl_svm',
 			  use_existing=True)
    metrics_list, pred_label = obj.predict(r'./mouse',
										  'mouse_chr19_test.csv',
											'dl_svm',
											r'./models/dl_svm',
											use_existing=True)

    print(np.unique(pred_label))	
	
