#!/usr/bin/env python3
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd#l
import anndata as ad
import scanpy as sc
import seaborn as sns
import pickle
import gc
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from torch.utils.data import Dataset, TensorDataset, DataLoader

#===============================================================================
#torch sppecific imports-------------------------------------------------------
#===============================================================================

from torch import optim
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from skorch.callbacks import EpochScoring
from skorch import NeuralNetRegressor, NeuralNet
import skorch 
import torch
import torch.nn as nn
import torch.nn.functional as F

# set seed 
import random
random.seed(42)

#===============================================================================
# setup model -------------------------------------------------------
#===============================================================================
n_genes = 100

# examples.
# https://github.com/probml/pyprobml/blob/4ede64b4cebf8a9c7fc40d5b14c5dd76c5667e50/scripts/mnist_skorch.py#L74
# https://github.com/spring-epfl/mia/blob/d389d30188ca115d21365f5b0595b1bb5f1bbee9/tests/test_estimators.py#L176
# https://github.com/fancompute/wavetorch/blob/56129e59b3171042316442b95ad4dc619255d299/study/vowel_train_sklearn.py#L86
# https://github.com/probml/pyprobml/blob/4ede64b4cebf8a9c7fc40d5b14c5dd76c5667e50/scripts/skorch_demo.py#L33
n_h=5000
print("n_h is:")
print(str(n_h))

class MyModule(nn.Module):
    def __init__(self, n_in = n_genes, n_h=n_h, n_layers=3, dropout=0.1):
        super().__init__()
        self.n_in = n_in
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(n_in, n_h))
        self.layers.append(nn.Dropout(dropout))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm1d(n_h))# after the RELU? like Michael Gee

        for _ in range(n_layers-1):
            self.layers.append(nn.Linear(n_h, n_h))
            self.layers.append(nn.Dropout(dropout))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm1d(n_h))

        # our output layer has no activation 
        # function as we're predicting a continuous variable
        self.layers.append(nn.Linear(n_h, 1)) 
    
    def forward(self, data, sample_weight):
        # when X is a dict, its keys are passed as kwargs to forward, thus
        # our forward has to have the arguments 'data' and 'sample_weight';
        # usually, sample_weight can be ignored here
        y = data.view(-1, self.n_in)
        for _, layer in enumerate(self.layers):
            y = layer(y)
        return y

    def get_loss(self, y_pred, y_true, X):
        # override get_loss to use the sample_weight from X
        loss_unreduced = super().get_loss(y_pred, y_true, X, *args, **kwargs)
        sample_weight = skorch.utils.to_tensor(X['sample_weight'], device=self.device)
        loss_reduced = (sample_weight * loss_unreduced).mean()
        return loss_reduced


class RegressionModule(NeuralNetRegressor):
    def __init__(self,*args, criterion__reduce=False, **kwargs):
        super().__init__(*args, criterion__reduce=criterion__reduce, **kwargs)

    def get_loss(self, y_pred, y_true, X, *args, **kwargs):
        # override get_loss to use the sample_weight from X
        loss_unreduced = super().get_loss(y_pred, y_true, X, *args, **kwargs)
        sample_weight = skorch.utils.to_tensor(X['sample_weight'], device=self.device)
        loss_reduced = (sample_weight * loss_unreduced).mean()
        return loss_reduced


#         self.n_in = n_in
#         self.layers = nn.ModuleList()
#         self.layers.append(nn.Linear(n_in, n_h))
#         self.layers.append(nn.Dropout(dropout))
#         self.layers.append(nn.ReLU())
#         self.layers.append(nn.BatchNorm1d(n_h))# after the RELU? like Michael Gee

#         for _ in range(n_layers-1):
#             self.layers.append(nn.Linear(n_h, n_h))
#             self.layers.append(nn.Dropout(dropout))
#             self.layers.append(nn.ReLU())
#             self.layers.append(nn.BatchNorm1d(n_h))

#         # our output layer has no activation 
#         # function as we're predicting a continuous variable
#         self.layers.append(nn.Linear(n_h, 1)) 

#     def forward(self, data, sample_weight):
#         # when X is a dict, its keys are passed as kwargs to forward, thus
#         # our forward has to have the arguments 'data' and 'sample_weight';
#         # usually, sample_weight can be ignored here
#         y = x.view(-1, self.n_in)
#         for _, layer in enumerate(self.layers):
#             y = layer(y)
#         return y

#     def get_loss(self, y_pred, y_true, X):
#         # override get_loss to use the sample_weight from X
#         loss_unreduced = super().get_loss(y_pred, y_true, X, *args, **kwargs)
#         sample_weight = skorch.utils.to_tensor(X['sample_weight'], device=self.device)
#         loss_reduced = (sample_weight * loss_unreduced).mean()
#         return loss_reduced


# class MyModule(nn.Module):
#     def __init__(self, n_in = n_genes, n_h=5000, n_layers=3, dropout=0.1,  *args, **kwargs):
#         super().__init__(*args, criterion__reduce=False, **kwargs)
#         self.n_in = n_in
#         self.layers = nn.ModuleList()
#         self.layers.append(nn.Linear(n_in, n_h))
#         self.layers.append(nn.Dropout(dropout))
#         self.layers.append(nn.ReLU())
#         self.layers.append(nn.BatchNorm1d(n_h))# after the RELU? like Michael Gee

#         for _ in range(n_layers-1):
#             self.layers.append(nn.Linear(n_h, n_h))
#             self.layers.append(nn.Dropout(dropout))
#             self.layers.append(nn.ReLU())
#             self.layers.append(nn.BatchNorm1d(n_h))

#         # our output layer has no activation 
#         # function as we're predicting a continuous variable
#         self.layers.append(nn.Linear(n_h, 1)) 

#     def forward(self, data, sample_weight):
#         # when X is a dict, its keys are passed as kwargs to forward, thus
#         # our forward has to have the arguments 'data' and 'sample_weight';
#         # usually, sample_weight can be ignored here
#         y = x.view(-1, self.n_in)
#         for _, layer in enumerate(self.layers):
#             y = layer(y)
#         return y

#     def get_loss(self, y_pred, y_true, X):
#         # override get_loss to use the sample_weight from X
#         loss_unreduced = super().get_loss(y_pred, y_true, X, *args, **kwargs)
#         sample_weight = skorch.utils.to_tensor(X['sample_weight'], device=self.device)
#         loss_reduced = (sample_weight * loss_unreduced).mean()
#         return loss_reduced

# # example is here: 
# class RegressionModule(NeuralNetRegressor):
#     def __init__(self, mod, criterion__reduce=False, *args, **kwargs):
#         # make sure to set reduce=False in your criterion, since we need the loss
#         # for each sample so that it can be weighted
#         super().__init__(module=mod, criterion__reduce=criterion__reduce,*args, **kwargs)

#     def forward(self, x, sample_weight):
        
        
#         y = x.view(-1, self.n_in)
#         for _, layer in enumerate(self.layers):
#             y = layer(y)
#         return y

# class MyNet(NeuralNet):
#     def __init__(self, *args, criterion__reduce=False, **kwargs):
#         # make sure to set reduce=False in your criterion, since we need the loss
#         # for each sample so that it can be weighted
#         super().__init__(*args, criterion__reduce=criterion__reduce, **kwargs)


#===============================================================================
# import data -------------------------------------------------------
#===============================================================================

### directories
base_dir = "/projects/pfenninggroup/singleCell/Ruzicka_snRNA_Seq"
data_dir = base_dir + "/data/"
out_dir = base_dir + "/results_standardized/neural_net/"
script_dir ="/home/qsu1/sc_age/scripts"

# the metadata of test and train.
train_df = pd.read_csv(data_dir+ "22_11_02_df_train.csv")
val_df = pd.read_csv(data_dir+ "22_11_02_df_val.csv")
train_df =pd.concat( [train_df, val_df], axis=0)
test_df = pd.read_csv(data_dir+ "22_11_02_df_test.csv")

# pass the sample_weight as a dictionary
weights = pd.read_csv(base_dir+"/results_standardized/glmnet/22_08_24_weights.csv")
temp=weights[["ID", "weight"]]
temp.index = temp["ID"]
weights_dict=temp["weight"].to_dict()



# pass the sample_weight as a dictionary
# weights = pd.read_csv(base_dir+"/results_standardized/glmnet/22_08_24_weights.csv")
# temp=weights[["ID", "weight"]]
# temp.index = temp["ID"]
# weights_dict=temp["weight"].to_dict()


#### import data
sc=ad.read_h5ad(data_dir+ "22_08_12_seurat_raw_FILTERED_neurons_only_controls_only.h5ad")
#sc_norm =ad.read_h5ad(data_dir+ "22_08_12_seurat_raw_FILTERED_neurons_only_controls_only.h5ad")

###### separate into test and train/val?
test = sc[sc.obs['ID'].isin(set(test_df['ID']))]
train = sc[sc.obs['ID'].isin(set(train_df['ID']))]

# plt.figure(figsize=(6,4))
# # sns.countplot( data= sc.obs,
# #                x = "Age")
# # the histogram of the data
# n, bins, patches = plt.hist(sc.obs["Age"], 12, density=True, facecolor='b', alpha=0.75)
# plt.xlabel('Age')
# plt.ylabel('Probability')
# plt.title("Histogram of target variable")
# plt.savefig(out_dir + "distribution.png")


###### scale the age
age_scaler = StandardScaler()
age_scaler.fit(sc.obs["Age"].values.reshape(-1, 1))
sc.obs["Age"]=age_scaler.transform(sc.obs["Age"].values.reshape(-1, 1))
# >>> np.max(sc.obs["Age"])
# 2.1942267495439243

###### separate into test and train/val?
cts_test = pd.read_pickle(base_dir + "/results_standardized/"+ "22_12_05svd_nonnorm_zscore_nmf_test.pkl")
# (26758, 100)

cts = pd.read_pickle(base_dir +"/results_standardized/"+"22_12_05svd_nonnorm_zscore_nmf_train.pkl")

# should technically only leave the sample_weight map back to the IDs. 
complete_weight = train.obs["ID"].map(weights_dict) # >>> (49, 6)
complete_weight=complete_weight.tolist()

#===============================================================================
# prediction task -------------------------------------------------------
#===============================================================================
datestring = "22_12_06" # for naming outputs.
LR_LIST = [1e-1, 1e-2, 1e-4]
LAYER_SIZES = [0.3]  # [1, 0.7, 0.5, 0.3, 0.1]
NUM_LAYERS = [3]
DROPOUT = [0, 0.5]
BATCH_NORM = [0, 1]
BATCH_SIZE = [8192, 4096, 2048, 1024]  # [32, 128, 1024, 4096]
OPTIMIZER = ["adam", "sgd", "sgd_mom90"]

print("BEGIN FITTING")
lr = LR_LIST[2]
#net = NeuralNetRegressor(MyModule,
net = RegressionModule(MyModule,
    lr = lr,
    batch_size = 4096,
    iterator_train__shuffle=True, 
    iterator_valid__shuffle=False,
    #optimizer=torch.optim.Adam(net.parameters(), lr=lr),
    optimizer__momentum=0.9,
    max_epochs=30, 
    # criterion__reduce=False, #  we need the loss for each sample so that it can be weighted
    #train_split = skorch.dataset.ValidSplit(5, stratified=False),
    train_split=skorch.dataset.ValidSplit(5, stratified=False))

# assemble data as tensors.
X=torch.tensor(cts).to(torch.float32)
X_test = torch.tensor(cts_test, dtype=torch.float32)
y= torch.from_numpy(train.obs['Age'].values.reshape(-1, 1)).float() # float32

# put data into a dict for weight purposes
X = {'data': X}
# add sample_weight to the X dict
X['sample_weight'] =complete_weight

net.fit(X,y)
# PICKLE FILE
with open(out_dir +datestring+'skorch_net_weighted_NMF_'+str(n_h)+'.pkl', 'wb') as f:
    pickle.dump(net, f)

net.save_params(f_params=out_dir +datestring+'_skorch_net_weighted_NMF_'+str(n_h)+'_params.pkl')

# model = pickle.load(out_dir +'22_11_18_skorch_net_weighted_50.pkl')
# model = pd.read_pickle(out_dir +'22_11_18_skorch_net_weighted_50.pkl')

train_pred =net.predict(X)

print("PASSED TRAINING")
#===============================================================================
# plot results -------------------------------------------------------
#===============================================================================

temp = pd.DataFrame(train.obs)
temp["predicted_age"] =age_scaler.inverse_transform(train_pred.reshape(-1, 1))
temp["Age"]= age_scaler.inverse_transform(train.obs["Age"].values.reshape(-1, 1))
temp.to_csv(out_dir +datestring+'_skorch_net_weighted_NMF_'+str(n_h)+"train_pred.csv")

# now for teh test set, theoretically no need for weights...why not?
# but skorch still calls for them, why?

# should technically only leave the sample_weight map back to the IDs. 
test_weight = test.obs["ID"].map(weights_dict) # >>> (49, 6)
#test_weight=test.tolist()

# put data into a dict for weight purposes
X = {'data': X_test}
# add sample_weight to the X dict
X['sample_weight'] =test_weight

y= torch.from_numpy(test.obs['Age'].values.reshape(-1, 1)).float()
test_pred = net.predict(X)
print("TEST PRED")
print(test_pred)

temp = pd.DataFrame(test.obs)
temp["predicted_age"] =age_scaler.inverse_transform(test_pred.reshape(-1, 1))
temp["Age"]= age_scaler.inverse_transform(test.obs["Age"].values.reshape(-1, 1))
temp.to_csv(out_dir +datestring+'_skorch_net_weighted_NMF_'+str(n_h)+"test_pred.csv")



