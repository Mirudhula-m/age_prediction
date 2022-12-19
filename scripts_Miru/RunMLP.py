"""
RUNNING FEEDFORWARD NEURAL NETWORK
This section saves the predicted test data and model library as a pickle.
Look for TestAnalysis.py for the analysis part

@author:Mirudhula Mukundan
@date:10/31/2022
@ID:mirudhum
"""

import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse as sp

from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn

from mlp import FNN

# fix random seed for reproducibility
seed = 7
torch.manual_seed(seed)


# get datavars
import pickle
with open('dataVars.pkl','rb') as file:
    [[trainX,trainY,trainID],[testX,testY,testID]] = pickle.load(file)
    

# Training Parameters
batch_size = 4096
N_epochs = 30
N_examples = trainX.shape[0]

# MLP parameters
inp_dim = trainX.shape[1]
out_dim = 1
hidden_dim = [5000,5000,5000]
dropout_prob = 0.4
eta = 0.001

# Get the model
model = FNN(inp_dim, out_dim, hidden_dim, dropout_prob) #inpd, outd, hidden, dropout_prob
model

# Training the model
lossfunc = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr = eta)
# optimizer = torch.optim.SGD(model.parameters(),lr = eta, momentum=0.9)
optimizer

# Some intializations
lossList = []
predictions = []
ground_truth = []
testLossList = []

scaler = StandardScaler(with_mean=False)
tX = scaler.fit_transform(trainX)
ttX = scaler.fit_transform(testX)

trainy = trainY.reshape(len(trainY), 1)
testy = testY.reshape(len(testY), 1)
scalerY = StandardScaler(with_mean=False)
scaleY = scalerY.fit(trainy)
tY = scalerY.transform(trainy).squeeze()
ttY = scalerY.transform(testy).squeeze()


# Training as epochs
# start_time = time.time()
model = model.double()
for epoch in range(N_epochs):
    
    loss = 0
    
    # Shuffling the training dataset
    perm = np.random.permutation(tX.shape[0])

    for batch in range(int(tX.shape[0]/batch_size)):

        shuff_idx = perm[batch*batch_size : (batch*batch_size+batch_size)]

        batch_x = tX[shuff_idx]
        batch_y = tY[shuff_idx]

        # convert to pytorch tensor
        x = torch.tensor(batch_x.toarray(), dtype=torch.float64)
        y = torch.tensor(batch_y)

        optimizer.zero_grad()
        y_pred = model(x).squeeze()

        batch_loss = lossfunc(y_pred, y) 
        loss += batch_loss.item()
        batch_loss.backward()
        optimizer.step()
        # if batch%10 == 0:
        #     print("Batch No.",batch+1,";Batch Loss =",batch_loss.item())
            
    ### Testing
    testx = torch.tensor(ttX.toarray(),dtype=torch.float64)
    testy = torch.tensor(ttY)
    test_pred = model(testx).squeeze()
    test_loss = lossfunc(test_pred, testy) 
        
    if epoch == N_epochs - 1:
        predictions.extend(test_pred.tolist())

    print(epoch+1,"train epochs completed =================== Loss =",loss/int(N_examples/batch_size))#tX.shape[0])
    print(epoch+1,"test epochs completed =================== Loss =",test_loss.item())
    


# Inverse transform all predictions
pred = np.round(scalerY.inverse_transform(np.array(predictions).reshape(len(predictions),1)))
gndt = scalerY.inverse_transform(np.array(ttY).reshape(len(predictions),1))

# store datavars
with open('predictions_b.pkl','wb') as file:
    pickle.dump([pred,gndt,testID],file)

torch.save(model.state_dict(), "age_model_b.pth")










