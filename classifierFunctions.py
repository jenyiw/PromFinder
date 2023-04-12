# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 14:35:50 2023

@author: Jenyi

Functions for creating and training the classifiers
"""

# from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def create_svm(X, Y):
	
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
	
	model = SVC()
	model.fit(X, Y)
	
	return model

def predict_svm(X, model):
	"""
	Make a prediction using trained model
	
	Parameters:
		X: numpy array
		 Matrix of (# samples, # features) for classification
		model
		 Trained model
		  
	Returns:
		predictions: numpy array
			Predictions of trained model
	"""		
	
	predictions = model.predict(X)
	
	return predictions

def metrics_svm(y_true, y_predict):
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
	
	metrics_list = [acc, f1, precision, recall]
	
	return metrics_list
	