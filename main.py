import os
import pickle
from turtle import forward 

import torch
import torch.nn as nn 
from torch.optim import Adam
import numpy as np
import pandas as pd  

from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set_style("whitegrid", {'axes.grid' : False})

from utils import viz 

#-------------------------
#     System folders
#-------------------------

# find the current path
path = os.path.dirname(os.path.abspath(__file__))
# create the folders for this folder
if not os.path.exists(f'{path}/figures'):
    os.mkdir(f'{path}/figures')

#--------------------------------
#        Hyperparameters
#--------------------------------

DefaultParams = { 
                'MaxEpochs': 15,
                'L2Reg': 1e-5,
                'SparsityReg': 5,
                'SparsityPro': .1,
                'Verbose': True,
                'BatchSize': 32,
                'If_gpu': False, 
                } 
eps_ = 1e-8

#-------------------------------
#      β-VAE Architecture
#-------------------------------

class bVAE( nn.Module):

    def __init__( self, dims, z_dim, gpu=False):
        '''β Variational Auto Encoder
    
        X --> Encoder --> mu, logvar
        e ~ N( 0, 1) 
        Z = e*mu + logvar
        Z --> Decoder --> X_hat  

        Inputs:
            dims: [input_dim, hidden_dims]
            z_dim: the dim for the bottleneck layer
            gpu: if we are going to use gpu 
        '''
        super().__init__()
        
        # construct encoder 
        self.encoder = nn.ModuleList()
        for i in range(len(dims)-1):
            self.encoder.append( nn.Linear( dims[i], dims[i+1]))
            self.encoder.append( nn.ReLU())
        self.mu = nn.Linear( dims[-1], z_dim)
        self.logvar = nn.Linear( dims[-1], z_dim)
        
        # construct decoder
        re_dims = list(reversed(dims+[z_dim]))
        self.decoder = nn.ModuleList()
        for i in range(len(re_dims)-2):
            self.decoder.append( nn.Linear( re_dims[i], re_dims[i+1]))
            self.decoder.append( nn.ReLU())
        self.decoder.append( nn.Linear( dims[-2], dims[-1]))
        self.decoder.append( nn.Sigmoid())
        
        # choose device 
        if gpu and torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'
        self.to( self.device)

    def forward( self, x):
        # Encoding 
        h      = self.encoder( x)
        mu     = self.mu( h)
        logvar = self.logvar( h)
        # Reparameterize 
        z      = self.reparam( mu, logvar)
        # Decoding
        x_hat  = self.decoder( z)
        return x_hat, mu, logvar 

    def reparam( self, mu, logvar):
        var = (.5*logvar).exp() 
        eps = torch.randn_like(var)
        z   = mu + var*eps 
        return z 

#-------------------------------
#       Loss function 
#-------------------------------

def loss_fn( x, x_hat, mu, logvar):
    
    
        
        





if __name__ == '__main__':

    print(1)