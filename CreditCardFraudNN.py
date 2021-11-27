import numpy as np
import pandas as pd
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
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix


'''class RemoteCache:
    def __init__(self):
        self.client =  base.Client(("local_host", 11211)) # client connection gets set up with default values for now
        
    def _query_server(self, data_id):
        pass
    def sampler(self):'''
        
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
    def __init__(self, *tensors):
        # set client for memcached
        # this sets the port to 11211 and also crucially adds a serializer
        self.client =  base.Client(("localhost", 11211), serde=serde.pickle_serde) # client connection gets set up with default values for now
        self.shadow_cache = set()
        self.tensors = tensors
        self.size = tensors[0].size(0)
        #print(tensors[])
        x = tuple(tensor[0] for tensor in self.tensors)

        # initially seed memcached server with X number of values
        for i in range(32):
            self._write_cache(i, [tensors[0][i].tolist(), tensors[1][i].tolist()])
            self.shadow_cache.add(i)
            #break
    def __getitem__(self, index):
        return self._query_cache(index)
    
    def __len__(self):
        return self.size
    
    def _query_cache(self, index):
        result = self.client.get(str(index))
        
        if result is None:
            key_to_get = str(random.sample(self.shadow_cache, 1)[0])
            #print(key_to_get)
            result = self.client.get(key_to_get)
            key_to_remove = random.sample(self.shadow_cache, 1)[0]
            #print(key_to_remove)
            self.shadow_cache.remove(key_to_remove)
            self.client.delete(str(key_to_remove))
            self._write_cache(index, [self.tensors[0][index].tolist(), self.tensors[1][index].tolist()])
            self.shadow_cache.add(index)
        # result should now be in the form of a list with data as first item and output as second

        item = tuple([torch.as_tensor(result[0]), torch.as_tensor(result[1])])

        return item
    
    def _write_cache(self, index, item):
        # update remote cache.  Ideally this should be done on the server
        # not the client, however that is a limitation of using memcached
        #print(index, item)
        self.client.set(str(index), item)

CSV_PATH = "data/creditcard.csv"

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
	num_epochs = 10
	minibatch_size = 32
	learning_rate = 1e-3
	
	cc_data = pd.read_csv(CSV_PATH)
	transactionData = cc_data.drop(['Time'], axis=1)
	transactionData['Amount'] = StandardScaler().fit_transform(transactionData['Amount'].values.reshape(-1, 1))
	
	model = binaryClassification()
	
	X = transactionData.drop("Class", axis=1).values
	y = transactionData['Class'].values
	
	X_tensor = torch.as_tensor(X)
	y_tensor = torch.as_tensor(y)
	
	X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=1)
	
	train_data = RemoteCacheDataset(X_train, y_train)
	test_data = TensorDataset(X_test)
	train_sampler = RemoteCacheSampler(train_data)
	train_loader = DataLoader(dataset=train_data, batch_size=minibatch_size,
                          sampler=train_sampler)
	
	test_loader = DataLoader(dataset=test_data, batch_size=1)
	
	criterion = nn.BCEWithLogitsLoss()
	optimizer = optim.Adam(model.parameters(), learning_rate)
	history = {}
	history['train_loss'] = []
	history['test_loss'] = []
	model.train()
	for e in range(1, num_epochs+1):
		print(e)
		epoch_loss = 0
		epoch_acc = 0
		for X_batch, y_batch in train_loader:
			#X_batch, y_batch = X_batch.to("cpu"), y_batch.to("cpu")
			optimizer.zero_grad()
			
			y_pred = model(X_batch)
			
			loss = criterion(y_pred, y_batch.unsqueeze(1).float())
			acc = binary_acc(y_pred, y_batch.unsqueeze(1))
			
			loss.backward()
			optimizer.step()
			
			epoch_loss += loss.item()
			epoch_acc += acc.item()
			

		print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')
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


train_model()
