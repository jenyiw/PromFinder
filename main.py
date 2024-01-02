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
import tkinter as tk


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
							 
        #Correct path names		  
        kmer_folder = os.path.join(folder_path, 'kmer_data')
        kmer_folder = kmer_folder.replace('\\', '/')
        cage_path = os.path.join(folder_path, 'cage_data')
        cage_path = cage_path.replace('\\', '/')
        genome_path = os.path.join(folder_path, 'genome_data')	
        genome_path = genome_path.replace('\\', '/')
        phylop_path = os.path.join(folder_path, 'phyloP_data')
        phylop_path = phylop_path.replace('\\', '/')
							 
        #get list of available chromosomes		
        chromosome_list = dF.get_chrom_list(genome_path)
							 
        #calculate features if features don't already exist			
        if not self.use_existing:
            kF.get_kmer_windows(genome_path, cage_path, cage_file, kmer_folder, window=self.kmer_size, train=train)
            fF.get_features(kmer_folder, window_size=self.kmer_size)

	#get features and labels
        feature_arr = dF.read_data(kmer_folder, chromosome_list, 'features_all')

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
				  
        #get model paths		
        model_path = os.path.join(os.path.dirname(train_folder), 'models', f'{classifier}')
        model_path = model_path.replace('\\', '/')
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        #get training data				  
        feature_arr, phylo_arr, label_arr = self.create_kmer_data(train_folder, cage_file_name)
				  
        #train and save classifiers		
        print(f'Training {classifier}...')			
        if classifier == 'dl':
	   
            train_data, train_label, val_data, val_label, train_phylo = dF.split_data(feature_arr, phylo_arr, label_arr, train_proportion=0.9, shuffle=True)
			
            cnnFunctions.create_CNN(model_path,
					   train_data, train_label.reshape(-1,1),
					   val_data[0:1,...], val_label[0:1,...].reshape(-1,1))
			

        elif classifier == 'svm':
			
            train_data = np.max(feature_arr, axis=1)

            save_model_path = os.path.join(model_path, 'svm_model.sav')
            save_model_path = save_model_path.replace('\\', '/')
            cF.create_svm(train_data, label_arr.reshape(-1,), save_model_path)			
		
        elif classifier == 'dl_svm': 
           train_data, train_label, val_data, val_label, train_phylo = dF.split_data(feature_arr, phylo_arr, label_arr, train_proportion=0.9, shuffle=True)

           cnnFunctions.create_CNN(model_path,
					   train_data, train_label.reshape(-1,1),
					   val_data[0:1,...], val_label[0:1,...].reshape(-1,1))
			
           pred_label, x_out = cnnFunctions.predict_CNN(model_path, train_data, train_label.reshape(-1,1),
												  feed_svm=True)


           save_model_path = os.path.join(model_path, 'svm_model.sav')
           save_model_path = save_model_path.replace('\\', '/')
           cF.create_svm(x_out, train_label.reshape(-1,), save_model_path)		

			
        elif classifier == 'rf':

             train_data = np.max(feature_arr, axis=1)			
             # train_data = np.concatenate((train_data, phylo_arr.reshape(-1,1)), axis=1)
             save_model_path = os.path.join(model_path, 'rf_model.sav')	
             save_model_path = save_model_path.replace('\\', '/')			
             cF.create_rf(train_data, label_arr.reshape(-1,), save_model_path)

        elif classifier == 'dl_rf': 
           train_data, train_label, val_data, val_label, train_phylo = dF.split_data(feature_arr, phylo_arr, label_arr, train_proportion=0.9, shuffle=True)

           cnnFunctions.create_CNN(model_path,
					   train_data, train_label.reshape(-1,1),
					   val_data[0:1,...], val_label[0:1,...].reshape(-1,1))
			
           pred_label, x_out = cnnFunctions.predict_CNN(model_path, train_data, train_label.reshape(-1,1),
												  feed_svm=True)

            #concatenate data
           # sum_data = np.max(train_data, axis=1)           			
           # pred_label = np.concatenate((pred_label.reshape(-1,1), sum_data), axis=1)
           save_model_path = os.path.join(model_path, 'rf_model.sav')
           save_model_path = save_model_path.replace('\\', '/')
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
        output_path = output_path.replace('\\', '/')
				  
        #get test data				  
        test_data, test_phylo, test_label = self.create_kmer_data(test_folder, cage_file_name, train=False)	
				  
        #Run classifiers on test data		
        print(f'Testing {classifier}...')	
		
        if classifier == 'svm':
            test_data = np.max(test_data, axis=1)		
            # test_data = np.concatenate((test_data, test_phylo.reshape(-1,1)), axis=1)		
            pred_label = cF.predict(test_data, os.path.join(model_path, 'svm_model.sav'))
            pred_label = pred_label.replace('\\', '/')
	
        elif classifier == 'rf':
            test_data = np.max(test_data, axis=1)			
            # test_data = np.concatenate((test_data, test_phylo.reshape(-1,1)), axis=1)	
            pred_label = cF.predict(test_data, os.path.join(model_path, 'rf_model.sav'))
            pred_label = pred_label.replace('\\', '/')		
	
        elif classifier == 'dl':
		
            pred_label, _ = cnnFunctions.predict_CNN(model_path, test_data, test_label.reshape(-1,1))

        elif classifier == 'dl_svm':
		
            prob, x_out = cnnFunctions.predict_CNN(model_path, test_data, test_label.reshape(-1,1))

			
           #run SVM
            save_model_path = os.path.join(model_path, 'svm_model.sav')
            save_model_path = save_model_path.replace('\\', '/')
            pred_label = cF.predict(x_out, save_model_path)


        elif classifier == 'dl_rf':
		
            prob, x_out = cnnFunctions.predict_CNN(model_path, test_data, test_label.reshape(-1,1))
			
	    #run rf
            save_model_path = os.path.join(model_path, 'rf_model.sav')
            save_model_path = save_model_path.replace('\\', '/')
            pred_label = cF.predict(x_out, save_model_path)
			
        metrics_list = cF.metrics(test_label, pred_label)
	
        for m in metrics_list.keys():	
            print(f'{m}: {metrics_list[m]:.2f}')
			
        df = pd.DataFrame({'label': list(pred_label)})
			
        #save final predictions		
        out_path = os.path.join(output_path, 'final_pred.csv')
        out_path = out_path.replace('\\', '/')
        df.to_csv(out_path)

        return metrics_list, pred_label

# if __name__ == "__main__":
#     # os.chdir('..')	
#     # obj = PromFinder(kmer_size=1000)
#     # obj.train(r'D:/CMU/semester2/03713/Bioinformatics Practicuum/03713_HMM_Model/human',
#  	# 		  'D:/CMU/semester2/03713/Bioinformatics Practicuum/03713_HMM_Model/human/cage_data/refTSS_v3.0_human_coordinate.hg38.bed',
#  	# 		  'D:/CMU/semester2/03713/Bioinformatics Practicuum/03713_HMM_Model/models/dl_svm',
#  	# 		  use_existing=True)
#     # metrics_list, pred_label = obj.predict(r'./mouse',
# 	# 									  'mouse_chr19_test.csv',
# 	# 										'dl_svm',
# 	# 										r'./models/dl_svm',
# 	# 										use_existing=True)

#     # print(pred_label)	
#     os.chdir('..')	
#     obj = PromFinder(kmer_size=1000)
#     obj.train('D:/CMU/semester2/03713/Bioinformatics Practicuum/03713_HMM_Model/human',
#  			  'refTSS_v3.0_human_coordinate.hg38.bed',
#  			  'dl_svm',
#  			  use_existing=True)
#     metrics_list, pred_label = obj.predict('D:/CMU/semester2/03713/Bioinformatics Practicuum/03713_HMM_Model/mouse',
# 										  'mouse_chr19_test.csv',
# 											'dl_svm',
# 											'D:/CMU/semester2/03713/Bioinformatics Practicuum/03713_HMM_Model/models/dl_svm',
# 											use_existing=True)

#     print(pred_label)	
#         # Generate the output string
    


def run_function():
    os.chdir('..')	
    obj = PromFinder(kmer_size=1000)
    obj.train('D:/CMU/semester2/03713/Bioinformatics Practicuum/03713_HMM_Model/human',
 			  'refTSS_v3.0_human_coordinate.hg38.bed',
 			  'dl_svm',
 			  use_existing=True)
    metrics_list, pred_label = obj.predict('D:/CMU/semester2/03713/Bioinformatics Practicuum/03713_HMM_Model/mouse',
										  'mouse_chr19_test.csv',
											'dl_svm',
											'D:/CMU/semester2/03713/Bioinformatics Practicuum/03713_HMM_Model/models/dl_svm',
											use_existing=True)

    
    # your function code here
    return pred_label

# Create a Tkinter window
root = tk.Tk()
root.title("My Python Script Output")

# Add a label to display the output
output_label = tk.Label(root, text="")
output_label.pack()

# Define a button that generates and displays the output
generate_button = tk.Button(root, text="Generate Output", command=lambda: output_label.config(text=run_function()))
generate_button.pack()

# Set the window position
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = int((screen_width - 600) / 2)
y = int((screen_height - 600) / 2)
root.geometry("600x600+{}+{}".format(x+50, y-50))

# Periodically update the output in the label
def update_output():
    output = run_function()
    output_label.config(text=output)
    root.after(1000, update_output)  # update every second

# Start updating the output
update_output()

# Start the Tkinter event loop
root.mainloop()
