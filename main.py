import os
import pickle
from turtle import forward 

import torch
import torch.nn as nn 
from torch.optim import Adam
import numpy as np
import pandas as pd  

from torchvision import datasets, transforms
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
                'Beta': 1,
                'LR': 1e-3,
                'MaxEpochs': 30,
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
        return x_hat, z, mu, logvar

    def get_latent( self, x):
        with torch.no_grad():
            h      = self.encoder( x)
            mu     = self.mu( h)
            logvar = self.logvar( h)
            # Reparameterize 
            z      = self.reparam( mu, logvar)
            return z 
    
    def decode( self, z):
        '''Decode for test only
        '''
        with torch.no_grad():
            return self.decoder(z)

    def reparam( self, mu, logvar):
        var = (.5*logvar).exp() 
        eps = torch.randn_like(var)
        z   = mu + var*eps 
        return z 

#-------------------------------
#       Loss function 
#-------------------------------

def lossFn( x, x_hat, mu, logvar, beta=1):

    recon_error = (x * x_hat.log() + (1-x)*(1-x_hat).log()).mean()
    kld = -.5 * ( 1 + logvar - mu.pow(2) - logvar.exp()).mean()
    return recon_error + beta * kld
    
#-------------------------------
#    Train beta Auto encoder 
#-------------------------------

def trainbVAE( train_data, model, **kwargs):
    '''Train a beta VAE

    Input:
        data: the data for training
        model: the AE model for training 
        LR: learning rate 
        beta: the level of constraint, beta>=0, the larger beta
            the larger constraint
        L2Reg: the weight for l2 norm
        SparsityReg: the weight for sparsity
        SparsityPro': the target level sparsity 
        MaxEpochs: maximum training epochs 
        BatchSize: the number of sample in a batch for the SGD
        Versbose: tracking the loss or ont 
    '''
    ## Prepare for the training 
    # set the hyper-parameter 
    HyperParams = DefaultParams
    for key in kwargs.keys():
        HyperParams[key] = kwargs[key]
    # preprocess the data
    x, y = train_data
    n_batch = int( len(x) / HyperParams['BatchSize'])
    x_tensor = x.type( torch.FloatTensor)
    y_tensor = y.type( torch.FloatTensor)
    _dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    _dataloader = torch.utils.data.DataLoader( _dataset, 
                batch_size=HyperParams['BatchSize'], drop_last=True)
    # decide optimizer 
    optimizer = Adam( model.parameters(), lr=HyperParams['LR'], 
                        weight_decay=HyperParams['L2Reg'])       
    ## get batch_size
    losses = []
    
    # start training
    model.train()
    for epoch in range( HyperParams['MaxEpochs']):

        ## train each batch 
        loss_ = 0        
        for _, (x_batch, _) in enumerate(_dataloader):
            # clear gradient 
            optimizer.zero_grad()
            # reshape the image
            x = torch.FloatTensor( x_batch).view( 
                x_batch.shape[0], -1).to( model.device)
            # reconstruct x
            x_hat, _, mu, logvar =  model.forward( x)
            # calculate the loss 
            loss = lossFn( x, x_hat, mu, logvar, 
                beta=HyperParams['beta'])
            # update
            loss.backward()
            optimizer.step()
            # store the te losses for plotting 
            loss_ += loss.detach().cpu().numpy() / n_batch

        # track training
        losses.append(loss_)
        if (epoch%1 ==0) and HyperParams['Verbose']:
            print( f'   Epoch:{epoch}, Loss:{loss_}')

    return model, losses 

#-------------------------------
#        Feature tuning 
#-------------------------------

def phi_tuning( latent, idx1, idx2, model, n_sample=10):

    # get two axies
    nr = nc = n_sample 
    xvec = np.linspace( -2, 2, 10)
    yvec = np.linspace( -2, 2, 10)
    fig, axs = plt.subplots( 10, 10, figsize=( 2.5*nc, 2.5*nr))
    for i, x in enumerate( xvec):
        for j, y in enumerate( yvec):
            ax = axs[ i, j]
            z = latent.copy()
            z[ 0, idx1] = x
            z[ 0, idx2] = y 
            x_hat = model.decode(z)
            ax.imshow( x_hat.item(), cmap='Greys')
    plt.tight_layout()
    plt.savefig( f'{path}/figures/feature tuning.png')

        
if __name__ == '__main__':

    ## Get Mnist data 
    mnist_data = datasets.MNIST('../data', train=True, download=True,
                                transform=transforms.Compose(
                                    [ transforms.ToTensor(), transforms.Normalize( (.1307,), (.3081,))]
                                ))
    data = (mnist_data.data.type( torch.FloatTensor) / 255).bernoulli()
    label = (mnist_data.targets.type( torch.FloatTensor) / 255).bernoulli()

    ## Compress 
    train = True 
    dims = [ 784, 400]
    z_dim = 200
    betas = [ 1, 30]

    for b in betas:

        if train:
            print( f'Train bVAE with beta={b}')
            model = bVAE( dims, z_dim, gpu=True)
            model, losses = trainbVAE( (data, label), model, if_gpu=True)
            torch.save( model.state_dict(), f'{path}/checkpts/mnist_model-beta={b}.pkl')
        else:
            ## Load a model. If no model, train one 
            model = bVAE( dims, 200)
            model.load_state_dict(torch.load(f'{path}/checkpts/mnist_model-beta={b}.pkl'))
    
        ## Visualize
        model.to('cpu') 
        model.eval()
        rng = np.random.RandomState( 2022)
        stim_idx = rng.choice( data.shape[0])
        img = data[ stim_idx, :, :]
        img_ten = torch.FloatTensor( img).view( 
                1, -1).to( model.device)
        latent  = model.get_latent( img_ten)
        idx1, idx2 = rng.choice( z_dim, size=2)
        phi_tuning( latent, idx1, idx2, model, n_sample=10) 

    