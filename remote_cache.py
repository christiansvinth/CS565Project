import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pymemcache.client import base
from pymemcache import serde
from torch.utils.data.sampler import Sampler
from torch.utils.data.dataset import Dataset

def connect_to_cache(host=None, serializer=None):
    '''
    Returns a connection instance to a memcached server
    '''
    if host is None:
        host = ("localhost", 11211)
    if serializer is None:
        serializer = serde.pickle_serde
    
    return base.Client(host, serde=serializer)

def write_cache(client, index, item):
       client.set(str(index), item)

def query_cache(client, index):
    if not isinstance(index, str):
        # this may fail if there is no direct conversion to a string
        index = str(index)
    result = client.get(index)

    return result

def remove_from_cache(client, index):
    client.delete(index)

if __name__ == "__main__":
    # just do a simple test, assuming connection to memcached can be made
    cc_data = pd.read_csv("creditcard.csv")
    transactionData = cc_data.drop(['Time'], axis=1)
    transactionData['Amount'] = StandardScaler().fit_transform(transactionData['Amount'].values.reshape(-1, 1))


    X = transactionData.drop("Class", axis=1).values
    y = transactionData['Class'].values

    X_tensor = torch.as_tensor(X)
    y_tensor = torch.as_tensor(y)

    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=1)

    # sanity check here:
    print("Initial values:")
    print(X_train[0], y_train[0])

    # Here be testing, first check connection
    client = connect_to_cache()
    print([X_train[0].tolist(), y_train[0].tolist()])
    write_cache(client, 0, [X_train[0].tolist(), y_train[0].tolist()])
    result = query_cache(client, 0)
    print("result = ", result)