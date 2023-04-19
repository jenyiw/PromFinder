# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 14:35:50 2023

@author: Jenyi

Functions for creating and training the classifiers
"""

# from sklearn.preprocessing import StandardScaler
import os
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier as RFC

def create_svm(X, Y, output_path):
	
	"""
	create SVM classifier
	
	Parameters:
		X: numpy array
		 Matrix of (# samples, # features) for classification
		Y: numpy array
		 Matrix of (# samples, ) containing sample labels	  
	Returns:
		label_arr: model
			Classification model
	"""	

	print('Training on SVMs!')
	model = SVC()
	model.fit(X, Y)
	
	#save model
	pickle.dump(model, open(output_path, 'wb'))	
	

def predict(X, model_path):
	"""
	Make a prediction using trained model
	
	Parameters:
		X: numpy array
		 Matrix of (# samples, # features) for classification
		model_path: str
		 Path to load trained model
		  
	Returns:
		predictions: numpy array
			Predictions of trained model
	"""		
	print('Predicting!')
	model = pickle.load(open(model_path, 'rb'))
	predictions = model.predict(X)
	
	return predictions

def create_rf(X, Y, output_path):
	
	"""
	create random forest classifier
	
	Parameters:
		X: numpy array
		 Matrix of (# samples, # features) for classification
		Y: numpy array
		 Matrix of (# samples, ) containing sample labels	  
	Returns:
		label_arr: model
			Classification model
	"""	

	model = RFC(n_estimators=100, max_depth=3)
	model.fit(X, Y)

	#save model
	pickle.dump(model, open(output_path, 'wb'))	
	


def metrics(y_true, y_predict):
	"""
	Calculate metrics
	
	Parameters:
		y_true: numpy array
		 Matrix of (# samples,) true labels
		y_pred: numpy array
		 Matrix of (# samples, ) predicted labels  
	Returns:
		metrics_list: list
			List containing evaluation scores
	"""		
	
	acc = accuracy_score(y_true, y_predict)
	f1 = f1_score(y_true, y_predict)
	precision = precision_score(y_true, y_predict)
	recall = recall_score(y_true, y_predict)
	
	metrics_dict = {'Accuracy': acc,
				 'F1': f1,
				 'Precision': precision,
				 'Recall': recall
				 }
	
	return metrics_dict
	