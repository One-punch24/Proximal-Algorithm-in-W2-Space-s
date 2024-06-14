import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import matplotlib.pyplot as plt
import copy
import sys

class Twolayer_NN(nn.Module):
    def __init__(self,d,m,weights,act='tanh'):
        super().__init__()
        self.m=m
        self.d=d
        if act=='tanh':
            self.activation=nn.Tanh()
        if weights is None: 
            self.weight=nn.Parameter(torch.randn(m,d))
            self.a=nn.Parameter(torch.randn(1,m))
            self.W=torch.cat((self.weight,self.a.T),dim=1)
        else:
            self.W=nn.Parameter(weights)
            self.weight=self.W[::,:d]
            print(self.weight.shape)
            self.a=self.W[::,d].reshape(1,m)
    def forward(self, input):
        # input shape should be (d,ndata)
        Wx=self.weight@input
        out=1.0/self.m*self.a@self.activation(Wx)
        return out
def Twolayer_NN_compute(weights,X):
    # weights: (number of neurons,d+1)
    d=weights.shape[1]-1
    m=weights.shape[0]
    W=weights[::,0:d]
    a=weights[::,d]
    Wx=nn.functional.tanh(W@X)
    out=1.0/m*a@Wx
    return out
class OTMap_NN(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.width=1000
        self.linear=nn.Linear(d,self.width)
        self.activation=nn.ReLU()
        self.out=nn.Linear(self.width,d)

    def forward(self, w):
        out_linear=self.linear(w)
        out_relu=self.activation(out_linear)
        out=self.out(out_relu)
        return out