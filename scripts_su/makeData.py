"""
PROCESSING OF DATA FROM THE ANNDATA FILE

@author:Mirudhula Mukundan, Qiao Su
@date:12/10/2022

@ID:qsu1
"""

import anndata
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse as sp

import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler

import pickle

### directories
base_dir = "/projects/pfenninggroup/singleCell/Ruzicka_snRNA_Seq"
data_dir = base_dir + "/data/"
out_dir = base_dir + "/results_standardized/"
script_dir ="/home/qsu1/sc_age/scripts"

print("Get Dependencies Complete.")
# Reading the data
data = anndata.read_h5ad(data_dir+ "22_08_12_seurat_raw_FILTERED_neurons_only_controls_only.h5ad")
dataX = data.X
print("h5ad data extranction complete.")

# Number of genes
N_features = data.shape[1]

# freeing up memory
data_red = data.obs[['ID','Age']]
data = data_red.copy()
del data_red

# get the splits from csv files
train_split = pd.read_csv(data_dir+ "22_11_02_df_train.csv")
test_split = pd.read_csv(data_dir+ "22_11_02_df_test.csv")
valid_split = pd.read_csv(data_dir+ "22_11_02_df_val.csv")

print("Get Train-Test splits complete.")

trainID = np.concatenate([train_split.ID, valid_split.ID])
testID = test_split.ID
del train_split
del test_split
del valid_split

# get training dataset
trainX = csr_matrix((0,N_features))
trainY = np.array([]).reshape(0)
train_rowID = np.array([]).reshape(0)
for p in range(len(trainID)):
    prows = dataX[data.ID == trainID[p]] # in csr format
    trainX = sp.vstack([trainX, prows])
    del prows
    
    row_Age = data[data.ID == trainID[p]].Age
    trainY = np.hstack([trainY, row_Age])
    del row_Age
    
    rowID = data[data.ID == trainID[p]].ID
    train_rowID = np.hstack([train_rowID, rowID])
    del rowID
trainX.shape, trainY.shape, train_rowID

print("Get Training Data Complete.")

# get testing dataset
testX = csr_matrix((0,N_features))
testY = np.array([]).reshape(0)
test_rowID = np.array([]).reshape(0)
for p in range(len(testID)):
    prows = dataX[data.ID == testID[p]] # in csr format
    testX = sp.vstack([testX, prows])
    del prows
    
    row_Age = data[data.ID == testID[p]].Age
    testY = np.hstack([testY, row_Age])
    del row_Age
    
    rowID = data[data.ID == testID[p]].ID
    test_rowID = np.hstack([test_rowID, rowID])
    del rowID
testX.shape, testY.shape, test_rowID

print("Get Testing Data Complete.")

print(trainX.shape)
print(testX.shape)

with open(data_dir+'dataVars_test.pkl','wb') as file:
    pickle.dump([[trainX,trainY,train_rowID],[testX,testY,test_rowID]],file)
