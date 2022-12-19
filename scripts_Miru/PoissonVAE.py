"""
POISSON VARIATIONAL AUTOENCODER

@author:Mirudhula Mukundan
@date:12/07/2022
@ID:mirudhum
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import pickle
from sklearn.preprocessing import StandardScaler
from torch.nn.functional import gumbel_softmax


torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


# Building Data Loader
class DataBuilder(object):
    def __init__(self, path):
        with open(path,'rb') as file:
            [[trainX,trainY,trainID],[testX,testY,testID]] = pickle.load(file)

        with open('dataVars_testonly.pkl','rb') as file:
            [testFinX, testFinY, testFinID] = pickle.load(file)

        print(trainX.shape,trainY.shape)
        self.trainX = torch.tensor(trainX.toarray(),dtype=torch.float64)
        self.trainY = torch.tensor(trainY,dtype=torch.float64)
        # self.testX = torch.tensor(testX.toarray(),dtype=torch.float64)
        # self.testY = torch.tensor(testY,dtype=torch.float64)
        self.testX = torch.tensor(testFinX.toarray(),dtype=torch.float64)
        self.testY = torch.tensor(testFinY,dtype=torch.float64)
        
        self.trainID = trainID
        # self.testID = testID
        self.testID = testFinID

    
    def GetTrainData(self):
        return [self.trainX, self.trainY]
    def GetTestData(self):
        return [self.testX, self.testY]
    def GetTestID(self):
        return self.testID
    def GetNumOfExamples(self):
        return self.trainX.shape[0]
    def GetNumOfFeatures(self):
        return self.trainX.shape[1]
    def Shuffle(self):
        self.perm = np.random.permutation(self.GetNumOfExamples())
    def GetMiniBatch(self, batch, batch_size):
        shuff_idx = self.perm[batch*batch_size : (batch*batch_size+batch_size)]
        return self.trainX[shuff_idx], self.trainY[shuff_idx]

    


# In[31]:


# Build the Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, inpd, hidden, latent_dim=3, y_dim=64):
        super(Autoencoder,self).__init__()
        
        # Encoder 
        layer = []
        start_i = inpd
        for i in hidden:
            layer.append(nn.Linear(start_i,i))
            layer.append(nn.BatchNorm1d(i))
            layer.append(nn.ReLU())
            # layer.append(nn.Dropout(0.4))
            start_i = i 
        # Adding latent vectors
        layer.append(nn.Linear(start_i,latent_dim))
        layer.append(nn.BatchNorm1d(latent_dim))
        
        self.encodingLayers = nn.Sequential(*layer)
        
        # Mu and sigma
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_sigma = nn.Linear(latent_dim, latent_dim)
        
        # reverse the hidden layer array
        hidden_rev = hidden[::-1]
        
        # Sampling vector
        layer = []
        layer.append(nn.Linear(latent_dim, latent_dim))
        layer.append(nn.ReLU())
        layer.append(nn.Linear(latent_dim, hidden_rev[0]))
        layer.append(nn.ReLU())
        
        self.samplingLayers = nn.Sequential(*layer)
        
        
        # Decoder - Reverse MLP
        layer = []
        start_i = hidden_rev[0]
        for i in hidden_rev:
            layer.append(nn.Linear(start_i,i))
            layer.append(nn.BatchNorm1d(i))
            layer.append(nn.ReLU())
            start_i = i
        # Adding input vectors
        layer.append(nn.Linear(start_i,inpd))
        
        self.decodingLayers = nn.Sequential(*layer)
        
    def encode(self, x):
        
        self.encode_out = self.encodingLayers(x)
        m = self.fc_mu(self.encode_out)
        v = self.fc_sigma(self.encode_out)
        return m, v
    
    def GetEncodeOutput(self):
        return self.encode_out
    
    def reparameterize(self, mu, var):
        if self.training:
            std = torch.sqrt(var + 1e-10)
            lambd = torch.randn_like(std) # prior
            return mu + lambd * std # l
        else:
            return mu
        
    def decode(self, z):
        # sampling
        samp_out = self.samplingLayers(z)
        decode_out = self.decodingLayers(samp_out)
        return decode_out
    
    def forward(self, x):
        # gaussian mix
        mu, var = self.encode(x)
        var = F.softplus(var)
        z = self.reparameterize(mu, var)
        return self.decode(z), mu, var
        

        



class VAELoss(nn.Module):
    def __init__(self):
        super(VAELoss, self).__init__()
        self.lossMSE = nn.MSELoss(reduction="sum")
        
    def forward(self, l_recon, x, mu, logvar, KL_weight):
        lossMSE = self.lossMSE(l_recon, x)
        lossKL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return lossMSE + KL_weight*lossKL
        
        
class Annealing(object):
    def __init__(self, weight, klstart, kl_annealtime, kl_fullscale):
        self.weight = weight
        self.klstart = klstart
        self.kl_annealtime = kl_annealtime
        self.kl_fullscale = kl_fullscale
        self.cycle = 0 # this is the cycle parameter that indicates whether weights are on or off
        self.cycleNum = 0
    def on_epoch_end (self, epoch):
        if epoch % self.klstart == 0 and epoch != 0:
            self.cycleNum += 1
            if self.cycle == 0:
                self.cycle = 1
            else:
                self.cycle = 0
        if self.cycle == 1:
            print(epoch, self.klstart,self.cycleNum,self.kl_annealtime)
            new_weight = min(self.weight + ((epoch - self.klstart*self.cycleNum)/ self.kl_annealtime), 1.)
            self.weight = new_weight
        else:
            self.weight = 0
        print("Current Weight is ",self.weight)
        return self.weight



class FNN(nn.Module):
    def __init__(self, inpd, outd, hidden, dropout_prob):
        super(FNN,self).__init__()
        
        layer = []
        start_i = inpd
        for i in hidden:
            layer.append(nn.Linear(start_i,i))
            layer.append(nn.BatchNorm1d(i))
            layer.append(nn.ReLU(inplace=True))
            layer.append(nn.Dropout(dropout_prob))
            start_i = i
        layer.append(nn.Linear(start_i,outd))
        
        self.layers = nn.Sequential(*layer)
        
    def forward(self, x):
        return self.layers(x)




def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)




# Get Data
data = DataBuilder('dataVars.pkl')




# Hyperparameters
inpd = data.GetNumOfFeatures()
hiddenLayers_Autoencoder = [300,150]#[800,500,150]
hiddenLayers_MLP = [20,20]
latentDim = 100
dropoutProb = 0.4
lr = 0.001
gamma = 0.997 # learning rate decay
# cosine annealing with warm reset
T_0 = 20


# Hyperparameters
N_epochs = 1000
batch_size = 4096
N_batches = data.GetNumOfExamples()/batch_size
# Some initializations
predictions = []
KL_weight = 0
pred_weight = 0

# The number of epochs at which KL loss should be included
klstart = 20
# number of epochs over which KL scaling is increased from 0 to 1
kl_annealtime = 10
# number of epochs it will run on full scale
kl_fullscale = 10






# Model, Optimizer, Loss
model = Autoencoder(inpd, hiddenLayers_Autoencoder, latentDim).to(device)
mlp = FNN(latentDim, 1, hiddenLayers_MLP, dropoutProb).to(device)
optimizer_adam = optim.SGD(model.parameters(), lr = lr)
optimizer = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer_adam, T_0 = T_0, verbose=True) #gamma=gamma) #ExponentialLR
optimizer_mlp = optim.Adam(mlp.parameters(), lr = 0.001)
VAElossFunc = VAELoss()
MSELossFunc = nn.MSELoss()
KL_annealing = Annealing(KL_weight, klstart, kl_annealtime, kl_fullscale)






print("Number of Features =",data.GetNumOfFeatures())
print("Number of Examples =", data.GetNumOfExamples())

# Training model
model = model.double()
mlp = mlp.double()
for epoch in range(N_epochs):
    model.train()
    loss = 0
    trainLoss = 0
    data.Shuffle()
    print("KL:")
    KL_weight = KL_annealing.on_epoch_end(epoch)


    for batch in range(int(N_batches)):
        x, y = data.GetMiniBatch(batch, batch_size)
        x = x.to(device)
        y = y.to(device)

        optimizer_adam.zero_grad()
        optimizer_mlp.zero_grad()
        recon_batch, mu, logvar = model(x)
        recon_batch = recon_batch.to(device)

        # supervised learning loss
        ypred = mlp(model.GetEncodeOutput()).squeeze()
        ypred = ypred.to(device)
        s_loss = MSELossFunc(ypred, y)

        # reconstruction loss
        batch_loss = VAElossFunc(recon_batch, x, mu, logvar, KL_weight)
        # batch_loss.backward()

        # total loss for this batch
        tot_batch_loss = batch_loss/batch_size + s_loss
        tot_batch_loss.backward()

        # total loss
        loss += batch_loss.item()/batch_size + s_loss.item()
        trainLoss += s_loss.item()

        # step
        optimizer_adam.step()
        optimizer_mlp.step()

    
        del ypred, x, y, recon_batch

        if batch%10 == 0:
            print("Batch No.",batch+1,";ED Loss =",batch_loss.item()/batch_size,"Pred Loss = ",s_loss.item())

        del batch_loss, s_loss, tot_batch_loss

    print(epoch+1,"train epochs completed ======== Loss =",trainLoss/N_batches)
    optimizer.step()

    # Get y predictions for Train-Validation set
    model.eval()
    testLoss = 0

    with torch.no_grad():
        x, y = data.GetTestData()
        x = x.to(device)
        y = y.to(device)
        recon_batch, mu, logvar = model(x)
        ypred = mlp(model.GetEncodeOutput()).squeeze()
        ypred = ypred.to(device)
        testLoss = MSELossFunc(ypred, y)
        del x, y, recon_batch

    print(epoch+1,"test epochs completed ======== Loss =",testLoss.item())

    if epoch == N_epochs - 1:
        predictions.extend(ypred.tolist())







pred = np.round(np.array(predictions))
_,y = data.GetTestData()
gndt = np.array(y)




with open('VAEpredictions_a.pkl','wb') as file:
    pickle.dump([pred,gndt,data.GetTestID()],file)

torch.save(model.state_dict(), "VAEage_model_a.pth")
torch.save(mlp.state_dict(), "VAEage_model_mlp.pth")






