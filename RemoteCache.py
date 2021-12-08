from pymemcache.client import base
from pymemcache import serde
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data.sampler import Sampler
from torch.utils.data.dataset import Dataset
from LRUCache import SimpleCacheQueue
import random
import heapq
    
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
        
        self.X = tensors[0]#[:10000]
        self.y = tensors[1]#[:10000]
        self.data_shape = self.X[0].shape
        print("X shape: ", self.X.shape)
        print("y shape: ", self.y.shape)
        print("Data Shape: ", self.data_shape)
        
        self.size = len(self.X)
        CACHE_SIZE = self.size // 4
        self.cache_size = CACHE_SIZE
        self.shadow_cache = SimpleCacheQueue(CACHE_SIZE)
        self.GLOBAL_CACHE_HITS = 0
        self.GLOBAL_CACHE_MISSES = 0
        self.DISK_LATENCY = .01
        #x = tuple(tensor[0] for tensor in self.tensors)
        self.eviction_method = eviction_method


        # importance sampling cache and necessary values
        self._data_variance = data_variance
        self._cache_variance = 0
        
        if self.eviction_method == 'importance':
            # use a standard list as a heap, and the heapq module to manage it
            # which gives direct control over the heap contentsf
            self.shadow_cache = []
            
            
        # initially seed memcached server with X number of values
        for i in range(CACHE_SIZE):
            self._write_cache(i, [self.X[i].tolist(), self.y[i].tolist()])
            if self.eviction_method == 'importance':
                heapq.heappush(self.shadow_cache, [1, i])
            else:
                self.shadow_cache.insert(i)

    def __getitem__(self, index):
        return self._query_cache(index)
    

    def __len__(self):
        return self.size
    
    def _query_cache(self, index):
    
        # If sample was already in cache
        result = self.client.get(str(index))
        

        if result is None:
            self.GLOBAL_CACHE_MISSES += 1
            if self.eviction_method == 'lru':
                # Fetch the data point "from disk"
                #time.sleep(self.DISK_LATENCY)
                
                self._write_cache(str(index), [self.X[index].tolist(), self.y[index].tolist()])
                self.shadow_cache.insert(index)
                result = self.client.get(str(index))
                assert result is not None

                key_to_remove = self.shadow_cache.evict()
                self.client.delete(str(key_to_remove))
                
            elif self.eviction_method == 'random-replace':
                #time.sleep(self.DISK_LATENCY*0.25) # Assume there will be some latency/overhead for the background fetching process
                replacement_sample_index = str(random.sample(list(self.shadow_cache.lookup), 1)[0])
                result = self.client.get(str(replacement_sample_index))
                
                if result is None:
                    result = [self.X[int(replacement_sample_index)], self.y[int(replacement_sample_index)]]
                
                self.shadow_cache.evict(replacement_sample_index)
                self.client.delete(str(replacement_sample_index))
                
                self._write_cache(index, [self.X[index], self.y[index]])
                self.shadow_cache.insert(index)
                
            elif self.eviction_method == 'never-evict':
                #time.sleep(self.DISK_LATENCY)
                # Directly pull the missing sample "from disk"
                result = [self.X[index], self.y[index]]

            elif self.eviction_method == 'importance':
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

        # result should now be in the form of a list with data as first item and output as second

        else:
            self.GLOBAL_CACHE_HITS += 1
            
            if self.eviction_method == 'importance':
                # need to update heap for shadow cache
                for item in self.shadow_cache:
                    if item[1] == index:
                        item[0] +=1
                        break
                heapq.heapify(self.shadow_cache)
            
        item = tuple([torch.as_tensor(result[0]), torch.as_tensor(result[1])])

        if not item[0].shape == self.data_shape:
            item = tuple([self.X[index], self.y[index]])
        
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
