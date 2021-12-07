import numpy as np
import pandas as pd
import argparse
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pymemcache.client import base
from pymemcache import serde
from torch.utils.data.sampler import Sampler
from torch.utils.data.dataset import Dataset
import random
import time
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from LRUCache import *
import heapq

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




class RemoteCacheSampler(Sampler):
    def __init__(self, dataset):
        # not efficient but keep copy of dataset in sampler
        self.dataset = dataset
        
        # create generator, which allows us to iterate over the dataset once and only once
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        self.generator=torch.Generator()
        self.generator.manual_seed(seed)
        
    def __iter__(self):
        # should return an iterator over dataset
        for g in torch.randperm(len(self.dataset), generator=self.generator).tolist():
            yield g#self.dataset[g]
            
    def __len__(self):
        # returns number of rows in dataframe
        return len(self.dataset)
    
class RemoteCacheDataset(Dataset):
    """
    The RemoteCacheDataset simulates a standard cache pulling data from disk, as the entirety
    of the dataset can successfully fit in memory. The RemoteCacheDataset maintains a small
    "shadow cache" which acts as a standard cache would on a system using a larger dataset.
    When a requested item is not in the shadow cache and must be "retreived from disk", a
    preset latency is added to retrieving that data item.
    """
    def __init__(self, *tensors, eviction_method='lru', data_variance=1):
        known_eviction_methods = ['random-replace', 'lru', 'never-evict', 'importance']
        
        
        if eviction_method not in known_eviction_methods:
            raise ValueError("Unknown method {}. Valid options are {}".format(eviction_method, known_eviction_methods))
        # set client for memcached
        # this sets the port to 11211 and also crucially adds a serializer
        self.client =  base.Client(("localhost", 11211), serde=serde.pickle_serde) # client connection gets set up with default values for now
        self.tensors = tensors
        self.size = tensors[0].size(0)
        CACHE_SIZE = len(self.tensors[0]) // 4
        self.shadow_cache = SimpleCacheQueue(CACHE_SIZE)
        self.GLOBAL_CACHE_HITS = 0
        self.GLOBAL_CACHE_MISSES = 0
        self.DISK_LATENCY = .01
        x = tuple(tensor[0] for tensor in self.tensors)
        self.eviction_method = eviction_method

        # importance sampling cache and necessary values
        self._data_variance = data_variance
        self._cache_variance = 0
        
        if self.eviction_method is 'importance':
            # use a standard list as a heap, and the heapq module to manage it
            # which gives direct control over the heap contentsf
            self.shadow_cache = []

        # initially seed memcached server with X number of values
        print("Dataset length: ", len(self.tensors[0]))
        print("Cache size: ", CACHE_SIZE)
        for i in range(CACHE_SIZE):
            self._write_cache(i, [tensors[0][i].tolist(), tensors[1][i].tolist()])

            if self.eviction_method is 'importance':
                heapq.heappush(self.shadow_cache, [1, i])
            else:
                self.shadow_cache.insert(i)
            
        print(len(self.shadow_cache.lookup))
    def __getitem__(self, index):
        return self._query_cache(index)
    

    def __len__(self):
        return self.size
    
    def _query_cache(self, index):
    
		# If sample was already in cache
        result = self.client.get(str(index))
        

        if result is None:
            if self.eviction_method == 'lru':
                # Fetch the data point "from disk"
                #time.sleep(self.DISK_LATENCY)
                self._write_cache(index, [self.tensors[0][index].tolist(), self.tensors[1][index].tolist()])
                self.shadow_cache.insert(index)
                result = self.client.get(str(index))
                assert result is not None

                key_to_remove = self.shadow_cache.evict()
                self.client.delete(str(key_to_remove))
                
            elif self.eviction_method == 'random-replace':
                #time.sleep(self.DISK_LATENCY*0.25) # Assume there will be some latency/overhead for the background fetching process
                replacement_sample_index = str(random.sample(list(self.shadow_cache.lookup), 1)[0])
                result = self.client.get(str(replacement_sample_index))
                
                self.shadow_cache.evict(replacement_sample_index)
                self.client.delete(str(replacement_sample_index))
                
                self._write_cache(index, [self.tensors[0][index].tolist(), self.tensors[1][index].tolist()])
                self.shadow_cache.insert(index)
                
            elif self.eviction_method == 'never-evict':
                #time.sleep(self.DISK_LATENCY)
                # Directly pull the missing sample "from disk"
                result = [self.tensors[0][index].tolist(), self.tensors[1][index].tolist()]
            
            elif self.eviction_method is 'importance':
                # if we get a cache miss we need to perform the following operations:
                #    1) select a random index in the queue
                #    2) update the counter of the selected index to be the largest in the queue
                #    3) evict LRU item and fetch the new item - should also increase its index so it won't
                #       immediately get evicted from the cache
                #    4) compute updated cache variance
                substitue_index = random.randint(0, len(self.shadow_cache) - 1)
                
                largest_value = self.shadow_cache[self.cache_size-1][0]
                self.shadow_cache[substitue_index][0] = largest_value + 1
                # re-heapify in case we selected the least recently used item
                heapq.heapify(self.shadow_cache)
                
                # get actual data from cache and scale - how to scale??
                key_to_get = str(self.shadow_cache[substitue_index][1])
                result = self.client.get(key_to_get)
                # scale result by simple multiplication: this scale factor will likely be small given the initial
                # variance of the overall dataset
                result = [(np.array(result[0]) * self._cache_variance/self._data_variance).tolist(), result[1]]
                
                #key_to_remove = random.sample(self.shadow_cache, 1)[0]
                # according to heapq, this is more efficient than explicitly doing a push/pop operation
                heapq.heappushpop(self.shadow_cache, [largest_value+2, index])
                self._write_cache(index, [self.tensors[0][index].tolist(), self.tensors[1][index].tolist()])
                
                # lastly recompute updated cache variance
                self._compute_cache_variance()
                    
                self.GLOBAL_CACHE_MISSES += 1
            
        # result should now be in the form of a list with data as first item and output as second

        else:
            self.GLOBAL_CACHE_HITS += 1

            if self.eviction_method is 'importance':
                # need to update heap for shadow cache
                for item in self.shadow_cache:
                    if item[1] == index:
                        item[0] +=1
                        break
                heapq.heapify(self.shadow_cache)
        item = tuple([torch.as_tensor(result[0]), torch.as_tensor(result[1])])

        return item
    
    def _write_cache(self, index, item):
        # update remote cache.  Ideally this should be done on the server
        # not the client, however that is a limitation of using memcached
        #print(index, item)
        self.client.set(str(index), item)
        
    def _compute_cache_variance(self):
        # compute the current variance of the cache
        # ideally this is something the cache server would do, however
        # again due to limitations of using memcached, we perform this on the client side
        temp_data = []
        # slow and inefficient way to compute variance, but also easist to proof out
        for i in range(self.cache_size):
            temp_data.append(self.tensors[0][i].tolist())

        temp_data = np.array(temp_data)
        self._cache_variance = np.var(temp_data)



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
    
    train_data = RemoteCacheDataset(X_train, y_train, eviction_method=args.eviction_method, data_variance=np.var(X))
    test_data = TensorDataset(X_test)
    train_sampler = RemoteCacheSampler(train_data)
    train_loader = DataLoader(dataset=train_data, batch_size=minibatch_size,
                          sampler=train_sampler)
    
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
		
		
		y_pred = model(X_batch)
		
		
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
    print(classification_report(y_test, y_pred_list))
    print(confusion_matrix(y_test, y_pred_list))

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


