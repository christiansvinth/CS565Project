import numpy as np
import pandas as pd
import argparse
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import Sampler
from torch.utils.data.dataset import Dataset
import random
import time
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from LRUCache import *
from RemoteCache import *

CSV_PATH = "data/creditcard.csv"



parser = argparse.ArgumentParser(description="Pytorch Profiled training on credit card fraud data")
parser.add_argument('--epochs', '-e', default=10, type=int, help='Number of epochs to run')
parser.add_argument('--suffix', default='0', type=str, help='Suffix for log file names')
parser.add_argument('--eval', action='store_true', default=False, help='Enable evaluation')
parser.add_argument('--batch_size', default=32, type=int, help='Batch Size for NN Training')
parser.add_argument('--log_path', default="", type=str, help="Path to directory for log files")
parser.add_argument('--eviction_method', default='lru', type=str, help="Eviction Strategy to use")
args = parser.parse_args()


#Profiling Setup

from profiler_utils import DataStallProfiler


GLOBAL_DATA_PROFILER = DataStallProfiler(args)
compute_time_list = []
data_time_list = []

class binaryClassification(nn.Module):
    def __init__(self):
        super(binaryClassification, self).__init__()
        self.layer_1 = nn.Linear(29, 64)
        self.layer_2 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        
        return x

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc
    
def train_model():
    # Initialize timing tools
    start_full = time.time()
    time_stat = []
    start = time.time()
    
    # Set Parameters and Create Model
    num_epochs = args.epochs
    minibatch_size = args.batch_size
    learning_rate = 1e-3
    
    model = binaryClassification()
    
    # Load data
    cc_data = pd.read_csv(CSV_PATH)
    transactionData = cc_data.drop(['Time'], axis=1)
    transactionData['Amount'] = StandardScaler().fit_transform(transactionData['Amount'].values.reshape(-1, 1))
    
    
    X = transactionData.drop("Class", axis=1).values
    y = transactionData['Class'].values
    
    X_tensor = torch.as_tensor(X)
    y_tensor = torch.as_tensor(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=1)
    
    train_data = RemoteCacheDataset(X_train, y_train, eviction_method=args.eviction_method)
    test_data = TensorDataset(X_test)
    train_loader = DataLoader(dataset=train_data, batch_size=minibatch_size)
    
    test_loader = DataLoader(dataset=test_data, batch_size=1)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), learning_rate)
    
    total_time = AverageMeter()
    
    for e in range(1, num_epochs+1):
        
        start_ep = time.time()
        
        avg_train_time = train_epoch(train_loader, model, criterion, optimizer, binary_acc, e)
        
        total_time.update(avg_train_time)
        
        dur_ep = time.time() - start_ep
        
        print("EPOCH DURATION + {}".format(dur_ep))
        time_stat.append(dur_ep)
	
    dur_full = time.time() - start_full
    print("Cache Hit Rate -- {}".format(train_data.GLOBAL_CACHE_HITS/(train_data.GLOBAL_CACHE_HITS+train_data.GLOBAL_CACHE_MISSES)))
    evaluate_model(model, test_loader, y_test)
    GLOBAL_DATA_PROFILER.stop_profiler()
	
    
		
	
def train_epoch(train_loader, model, criterion, optimizer, accuracy_function, e):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	
	model.train()
	end = time.time()
	GLOBAL_DATA_PROFILER.start_data_tick()
	dataset_time = compute_time = 0
	
	epoch_loss = 0
	epoch_acc = 0
	for X_batch, y_batch in train_loader:
		# Compute data loading time
		data_time.update(time.time() - end)
		dataset_time += (time.time() - end)
		compute_start = time.time()
		
		# Switch to tracking compute time
		GLOBAL_DATA_PROFILER.stop_data_tick()
		GLOBAL_DATA_PROFILER.start_compute_tick()
		
		
		y_pred = model(X_batch.float())
		
		
		loss = criterion(y_pred, y_batch.unsqueeze(1).float())
		acc = accuracy_function(y_pred, y_batch.unsqueeze(1))
		
		optimizer.zero_grad()
				
		loss.backward()
		optimizer.step()
		
		epoch_loss += loss.item()
		epoch_acc += acc.item()
		
		# Swtich to tracking data loading time
		GLOBAL_DATA_PROFILER.stop_compute_tick()
		GLOBAL_DATA_PROFILER.start_data_tick()
		
		compute_time += (time.time() - compute_start)
		batch_time.update(time.time() - end)
		
		end = time.time()
	
	data_time_list.append(dataset_time)
	compute_time_list.append(compute_time)
	print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')
	
	return batch_time.avg

	
    
def evaluate_model(model, test_loader, y_test):
    y_pred_list = []
    model.eval()
    with torch.no_grad():
        for X_batch in test_loader:
            #print(X_batch)
            y_test_pred = model(X_batch[0].float())
            y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_tag = torch.round(y_test_pred)
            y_pred_list.append(y_pred_tag.cpu().numpy())
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    CR = classification_report(y_test, y_pred_list, output_dict=True)
    print(classification_report(y_test, y_pred_list))
    print(confusion_matrix(y_test, y_pred_list))
    print(CR['accuracy'])
    

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


train_model()


