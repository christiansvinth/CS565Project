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
from torchvision import datasets, transforms
import random
import time
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import classification_report, confusion_matrix
from LRUCache import *
from RemoteCache import *


parser = argparse.ArgumentParser(description="Pytorch Profiled training on credit card fraud data")
parser.add_argument('--epochs', '-e', default=10, type=int, help='Number of epochs to run')
parser.add_argument('--suffix', default='0', type=str, help='Suffix for log file names')
parser.add_argument('--eval', action='store_true', default=False, help='Enable evaluation')
parser.add_argument('--batch_size', default=64, type=int, help='Batch Size for NN Training')
parser.add_argument('--log_path', default="", type=str, help="Path to directory for log files")
parser.add_argument('--eviction_method', default='lru', type=str, help="Eviction Strategy to use")
args = parser.parse_args()


#Profiling Setup

from profiler_utils import DataStallProfiler


GLOBAL_DATA_PROFILER = DataStallProfiler(args)
compute_time_list = []
data_time_list = []


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    
def train_model():
    # Initialize timing tools
    start_full = time.time()
    time_stat = []
    start = time.time()
    
    transform=transforms.Compose([
        transforms.Resize((28,28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    # Load data
    train_dataset = datasets.MNIST('./data', train=True, download=True,
                                    transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True,
                                    transform=transform)
    X_train, y_train = train_dataset.data, train_dataset.targets
    X_test, y_test = test_dataset.data, test_dataset.targets
    
    train_data = RemoteCacheDataset(X_train, y_train, eviction_method=args.eviction_method)

    #train_sampler = RemoteCacheSampler(train_data)
    
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size)
    
    #test_dataset = Dataset(X_test, y_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1)
    
    
    # Set Parameters and Create Model
    learning_rate = 1
    
    model = ConvNet()
    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    
    total_time = AverageMeter()
    
    for e in range(1, args.epochs+1):
        print("ENTERING EPOCH ", e)
        start_ep = time.time()
        
        avg_train_time = train_epoch(train_loader, model, optimizer, e)
        
        total_time.update(avg_train_time)
        
        dur_ep = time.time() - start_ep
        scheduler.step()
        print("EPOCH DURATION + {}".format(dur_ep))
        time_stat.append(dur_ep)
	
    dur_full = time.time() - start_full
    print("Cache Hit Rate -- {}".format(train_data.GLOBAL_CACHE_HITS/(train_data.GLOBAL_CACHE_HITS+train_data.GLOBAL_CACHE_MISSES)))
    
    evaluate_model(model, test_loader, y_test)
    GLOBAL_DATA_PROFILER.stop_profiler()
	
    
		
	
def train_epoch(train_loader, model, optimizer, e):
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
        optimizer.zero_grad()
        output = model(X_batch.unsqueeze(1).float())
        loss = F.nll_loss(output, y_batch)
        loss.backward()
        optimizer.step()
        
        # Swtich to tracking data loading time
        GLOBAL_DATA_PROFILER.stop_compute_tick()
        GLOBAL_DATA_PROFILER.start_data_tick()
        
        compute_time += (time.time() - compute_start)
        batch_time.update(time.time() - end)
        
        end = time.time()
	
    data_time_list.append(dataset_time)
    compute_time_list.append(compute_time)
    
    return batch_time.avg

	
    
def evaluate_model(model, test_loader, y_test):
    y_pred_list = []
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            output = model(X_batch)
            y_test_pred = output.argmax(dim=1, keepdim=True)
            y_pred_list.append(y_test_pred)
            
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    print(y_test[:10])
    print(y_pred_list[:10])
    CR = classification_report(y_test, y_pred_list, output_dict=True)
    print(CR)
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


