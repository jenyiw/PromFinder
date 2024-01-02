# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 14:35:50 2023

@author: Jenyi

Functions for creating and training the classifiers
"""

import os
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier as RFC
import matplotlib.pyplot as plt

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

	model = RFC(n_estimators=100,)
	model.fit(X, Y)

	#save model
	pickle.dump(model, open(output_path, 'wb'))	
	


def metrics(y_true, y_predict):
	"""
	Calculate the following metrics: accuracy, F1, precision, recall and plot the confusion matrix
	
	Parameters:
		y_true: numpy array
		 Matrix of (# samples,) true labels
		y_pred: numpy array
		 Matrix of (# samples, ) predicted labels  
	Returns:
		metrics_list: list
			List containing evaluation scores
	"""		
	
	y_int_pred = [int(x) for x in y_predict]
	y_int_true = [int(x) for x in y_true]
	
	y_predict = y_int_pred
	y_true = y_int_true
	
	acc = accuracy_score(y_true, y_predict)
	f1 = f1_score(y_true, y_predict)
	precision = precision_score(y_true, y_predict)
	recall = recall_score(y_true, y_predict)
	
	conf_mat = confusion_matrix(y_true, y_predict)
	
	plt.imshow(conf_mat)
	plt.show()
	
	metrics_dict = {'Accuracy': acc,
				 'F1': f1,
				 'Precision': precision,
				 'Recall': recall
				 }
	
	return metrics_dict
	
