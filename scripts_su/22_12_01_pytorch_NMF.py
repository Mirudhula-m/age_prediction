#!/usr/bin/env python3
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import seaborn as sns
import pickle
import gc
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF

#from Allfuncs import *

# set seed 
import random
random.seed(42)


#===============================================================================
# import data -------------------------------------------------------
#===============================================================================

### directories
base_dir = "/projects/pfenninggroup/singleCell/Ruzicka_snRNA_Seq"
data_dir = base_dir + "/data/"
out_dir = base_dir + "/results_standardized/"
script_dir ="/home/qsu1/sc_age/scripts"

# the metadata of test and train.
train_df = pd.read_csv(data_dir+ "22_11_02_df_train.csv")
val_df = pd.read_csv(data_dir+ "22_11_02_df_val.csv")
train_df =pd.concat( [train_df, val_df], axis=0)
test_df = pd.read_csv(data_dir+ "22_11_02_df_test.csv")


#### import data
sc=ad.read_h5ad(data_dir+ "22_08_12_seurat_raw_FILTERED_neurons_only_controls_only.h5ad")
#sc_norm =ad.read_h5ad(data_dir+ "22_08_12_seurat_raw_FILTERED_neurons_only_controls_only.h5ad")

###### separate into test and train/val?
test = sc[sc.obs['ID'].isin(set(test_df['ID']))]
train = sc[sc.obs['ID'].isin(set(train_df['ID']))]

#===============================================================================
# z-score vs. not z-score data --------------------------------
# log the matrix after z-score vs. not log --------------------------------
#===============================================================================
norm_applied = [True, False]
z_score = [True, False]
log_applied = [True, False]
scaler = StandardScaler(with_mean=False)

# a loop to generate the final processed data.
def process_data(data, datestr, norm=False, z_score=False):
    # which dataset
    name =datestr+ "svd"
    if norm:
        name = name+"_norm"
    else:
        name = name+"_nonnorm"

    # apply z-scoring.
    if z_score:
        cts= scaler.fit_transform(data.X).astype(np.float64) 
        name = name+"_zscore"
        return name, cts
    else:
        return name, data.X

# note that we fit the PCA on the entire dataset as its unsupervised.
date = "22_12_05"
norm = False
zscore=True
name, cts =process_data(train, date, norm, zscore)

#===============================================================================
# perform PCA + check correlation with Age.--------------------------------
#===============================================================================
# pca = PCA(n_components=1000)
# pca.fit(cts)
# X_reduced = pca.fit_transform(cts) # data in form (n_samples, n_features)

model = NMF(n_components=100, init='random', random_state=0)
X_reduced= model.fit_transform(cts)
X_test = model.transform(test.X)


with open(out_dir+name+'_nmf_train.pkl', 'wb') as file:
    pickle.dump(X_reduced, file)

with open(out_dir+name+'_nmf_test.pkl', 'wb') as file:
    pickle.dump(X_test, file)

# with open(out_dir+'svd_1000.pkl', 'rb') as handle:
#     X_reduced = pickle.load(handle)


# visualize the first 3 components of PCA first..
fig = plt.figure(1, figsize=(12, 10))
ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)

ax.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    X_reduced[:, 2],
    #c=sc_norm.obs["Age"],
    #cmap=plt.cm.Set1,
    edgecolor="k",
    s=40,
)

ax.set_title(name)
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])
fig.savefig(out_dir +name+'_1000.png')

