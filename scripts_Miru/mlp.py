"""
THIS IS A SIMPLE MULTI-LAYER PERCEPTRON
Each layer has 1 linear, 1 ReLU activation function,
1 Batch Normalization function between layers, 1 Dropout function.
The output layer is linear as well.

@author:Mirudhula Mukundan
@date:10/31/2022
@ID:mirudhum
"""

# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import torch.nn as nn


class FNN(nn.Module):
    def __init__(self, inpd, outd, hidden, dropout_prob):
        super(FNN,self).__init__()
        
        layer = []
        start_i = inpd
        for i in hidden:
            layer.append(nn.Linear(start_i,i))
            layer.append(nn.ReLU(inplace=True))
            layer.append(nn.BatchNorm1d(i))
            layer.append(nn.Dropout(dropout_prob))
            start_i = i
        layer.append(nn.Linear(start_i,outd))
        
        self.layers = nn.Sequential(*layer)
        
    def forward(self, x):
        return self.layers(x)

