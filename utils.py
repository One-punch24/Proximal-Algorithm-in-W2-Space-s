import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import matplotlib.pyplot as plt
import copy
import sys
from scipy.spatial import cKDTree
from scipy.special import gamma
from nn_structure import *

def generate_data(opt):
    # Generate Label
    X_train=torch.randn(opt.d,opt.ndata)
    beta=torch.randn(1,opt.d)
    y_train_truth=beta@X_train
    # X_test=torch.randn(opt.d,opt.ndata)
    # y_test_truth=beta@X_test
    return X_train,y_train_truth

def optimize_nn_withOT(opt, X,y_truth,weights,noise_gd=False):
    OTMao_NN_array=[]
    train_losses=[]
    neg_entropies=[]
    for i in range(opt.steps_eta):
        otmap_nn=OTMap_NN(opt.d+1)
        def func(weights):
            return otmap_nn(weights)
        optimizer=optim.SGD(otmap_nn.parameters(),lr=opt.lr_otmap)
        otmap_nn.train()
        y_pred=Twolayer_NN_compute(weights,X).unsqueeze(0)
        nn_loss=nn.functional.mse_loss(y_pred,y_truth)
        print("Steps_eta ",i, "--- NN Loss", nn_loss)
        train_losses.append(nn_loss.detach().cpu().numpy().item())

        if noise_gd:
            weights_numpy=weights.detach().cpu().numpy() #(m,d+1)
            neg_entropy=-1.0*estimate_entropy(weights_numpy)
            neg_entropies.append(neg_entropy)
        # loss=0.0
        for j in range(opt.iters_lr):
            optimizer.zero_grad()
            weights_after_map=otmap_nn(weights)
            y_pred=Twolayer_NN_compute(weights_after_map,X)
            y_pred=y_pred.unsqueeze(0)
            if noise_gd:
                jacob_det_abs_log=0.0
                for i in range(opt.m):
                    jacob=torch.autograd.functional.jacobian(func,weights[i],create_graph=True)
                    jacob_det=torch.det(jacob)
                    jacob_det_abs_log+= torch.log(torch.abs(jacob_det))
                jacob_det_abs_log=torch.log(torch.abs(jacob_det))*1.0/opt.m 
                # print(jacob_det_abs_log)  
                loss=opt.tau*jacob_det_abs_log +nn.functional.mse_loss(y_pred,y_truth)+nn.functional.mse_loss(weights_after_map,weights)/(2.*opt.eta)
            else:
                loss=nn.functional.mse_loss(y_pred,y_truth)+nn.functional.mse_loss(weights_after_map,weights)/(2.*opt.eta)
            if j%opt.print_freq==0:
                print("Total loss",loss)
            loss.backward()
            optimizer.step()
        weights=otmap_nn(weights).clone().detach()
    return train_losses,neg_entropies

def optimize_nn(opt,X_train,y_train_truth,weights,noise_gd=False):
    model=Twolayer_NN(opt.d,opt.m,weights)
    optimizer = optim.SGD(model.parameters(), lr=opt.eta*opt.m)
    criterion=nn.MSELoss()
    model.train()
    train_losses=[]
    neg_entropies=[]
    for i in range(opt.steps_eta):
        # Training
        optimizer.zero_grad()
        y_train_pred= model(X_train)
        train_loss = criterion(y_train_pred,y_train_truth)
        train_losses.append(train_loss.detach().cpu().numpy().item())
        train_loss.backward()

        # Noisy Gradient Descent
        if noise_gd:
            for param in model.parameters():
                if param.grad is not None:
                    noise = torch.randn_like(param.grad) * np.sqrt(2*opt.tau*opt.eta)
                    param.grad += noise
            weights_numpy=model.W.detach().cpu().numpy() #(m,d+1)
            neg_entropy=-1.0*estimate_entropy(weights_numpy)
            neg_entropies.append(neg_entropy)
        optimizer.step()

        
    return train_losses,neg_entropies

def estimate_entropy(samples):
    """
    1-nearest neighbor: sample: (m,d) 
    """
    m,d=samples.shape
    tree=cKDTree(samples)
    
    # Find the nearest neighbor
    distances, _=tree.query(samples,k=2)
    nearest_distances=distances[:,1]  # The second is the distance of nearest neighbor
    
    # Computation of Entropy
    gamma_const = 0.57721566490153286060  # Euler Const
    volume_unit_ball = np.pi**(d/2) / gamma(d/2+1)  
    entropy_estimate = (d/m) * np.sum(np.log(nearest_distances)) + np.log(m - 1) + gamma_const + np.log(volume_unit_ball)
    
    return entropy_estimate

if __name__=='main':
    samples=np.array([[0,0],[0,1]])
    estimated_entropy = estimate_entropy(samples)
    print(estimated_entropy) # theoretically should be 0.5772+np.log(3.14159), which is correct.