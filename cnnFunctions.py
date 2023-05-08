# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 11:50:07 2023

@author: Asus
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import os


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(7,16,15)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16,64,5)
        # self.fc1 = nn.Linear(7616, 10)
        self.fc1 = nn.Linear(15616, 10)		
        self.fc2 = nn.Linear(10, 1)
        self.batchn1 = nn.BatchNorm1d(16)
        self.batchn2 = nn.BatchNorm1d(64)
        self.drop = nn.Dropout(0.1)
		
    def forward(self, x):
        x = self.pool(self.batchn1(F.relu(self.conv1(x))))
        x = self.drop(x)
        x = self.pool(self.batchn2(F.relu(self.conv2(x))))	
        x = self.drop(x)		
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        # self.fc1 = nn.Linear(x.shape[1], 10)	
        x = self.fc1(x)		
        x = self.fc2(x)		
        x = F.sigmoid(x)		
        return x


class hybrid_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(7,16,15)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16,64,5)
        # self.fc1 = nn.Linear(7616, 10)
        self.fc1 = nn.Linear(15616, 10)		
        self.fc2 = nn.Linear(10, 1)
        self.batchn1 = nn.BatchNorm1d(16)
        self.batchn2 = nn.BatchNorm1d(64)
        self.drop = nn.Dropout(0.1)
		
    def forward(self, x):
        x = self.pool(self.batchn1(F.relu(self.conv1(x))))
        x = self.drop(x)
        x = self.pool(self.batchn2(F.relu(self.conv2(x))))	
        x = self.drop(x)		
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        # self.fc1 = nn.Linear(x.shape[1], 10)	
        x_out = self.fc1(x)		
        x = self.fc2(x_out)		
        x = F.sigmoid(x)		
        return x, x_out


def accuracy(pred_y, y):
	    """Calculate accuracy."""
	    y = np.rint(y.detach().numpy())

	    return np.sum(pred_y == y) / len(y)
	
def data_transform(train_data, train_label, batch_size:int=10, shuffle:bool=True):
	
    train_data = np.moveaxis(train_data, -1, 1)
    train_data = torch.from_numpy(train_data).float()
    train_label = torch.from_numpy(train_label).float()
    train_dataset = TensorDataset(train_data, train_label)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader


def test_model(model, loader,
			   batch_size, 
			   validation:bool=False,
			   feed_svm:bool=False):

    criterion = nn.BCELoss()

    model.eval()
    total_loss = 0
    acc = 0
    pred_list = np.zeros((len(loader)))
    fl_list = []

    with torch.no_grad():
        for i, data in enumerate(loader):
			
            inputs, labels = data
			
            predicted, x_out = model(inputs)
				
            fl_list.append(x_out.detach().numpy())
            loss = criterion(predicted, labels)
            total_loss += loss / len(loader)
            predicted_np = predicted.detach().numpy()
            predicted_np[predicted_np >= 0.5] = 1
            predicted_np[predicted_np != 1] = 0
            acc += accuracy(predicted_np, labels) / len(loader)
			
            if not validation:
                if not feed_svm:
                    pred_list[i] = predicted_np
                else:
                    pred_list[i] = predicted.detach().numpy()


    model.train()

    return total_loss, acc, pred_list, fl_list


def train_model(model, train_loader, val_loader):
    criterion = nn.BCELoss()

    model.train()

    optimizer = torch.optim.Adam(model.parameters(),)

    for epoch in range(10):

        total_loss = 0
        acc = 0
        total_val_loss = 0
        total_val_acc = 0		
	
        for i, data in enumerate(train_loader):
		
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
		
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            predicted, x_out = model(inputs)     
            loss = criterion(predicted, labels)
			  
            #Record losses
            total_loss += loss / len(train_loader)
            predicted_np = np.rint(predicted.detach().numpy())
            acc += accuracy(predicted_np, labels) / len(train_loader)	    
			  
            loss.backward()
            optimizer.step()
		
            # Validation
            val_loss, val_acc, _, _ = test_model(model, val_loader, 1, validation=True)
            total_val_loss += val_loss
            total_val_acc += val_acc		

        # print statistics
        # Print metrics every 5 epochs
        if((epoch+1) % 5 == 0):
            print(f'Epoch {epoch+1:>3} | Train Loss: {total_loss:.2f} '
                            f'| Train Acc: {acc*100:>5.2f}% '
                            f'| Val Loss: {val_loss:.2f} '
                            f'| Val Acc: {val_acc*100:.2f}%'
                            # f'| Time: {int(time.time() - time_curr)}'
							)

    print('Finished Training')
    return model

def save_model(model, save_path):		
	            
	   torch.save(model.state_dict(), os.path.join(save_path, 'model'))

def create_CNN(save_path,
			   train_data, train_label,
			   val_data, val_label):
	
    print('Training model!')

    train_loader = data_transform(train_data, train_label)
    val_loader = data_transform(val_data, val_label)

    model = hybrid_CNN()
    model = train_model(model, train_loader, val_loader)

    save_model(model, save_path)
	
def predict_CNN(save_path, test_data, test_label, feed_svm:bool=False):
	
    if feed_svm == False:
        print('Testing model')
    test_loader = data_transform(test_data, test_label, batch_size=1, shuffle=False)
	
    model = hybrid_CNN()
    model.load_state_dict(torch.load(os.path.join(save_path, 'model')))

    model.eval()	
    loss, acc, predicted, x_out_list = test_model(model, test_loader, 1, feed_svm=feed_svm)
    x_out_list = np.concatenate(x_out_list, axis=0)
	
    print(f'CNN Loss: {loss:.2f}',
	      f'CNN Accuracy: {acc:.2f}')
	
    predicted = predicted.reshape(-1)
	
    return predicted, x_out_list
	
	

