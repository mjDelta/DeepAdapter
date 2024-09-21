import torch
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from functools import partial
from torch import nn, optim

def weight_init(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.02)
        m.bias.data.zero_()

def net_param(net):
    param = sum([p.numel() for p in net.parameters()])
    return param / 1e6

class AE(nn.Module):
    """docstring for AE"""
    def __init__(self, in_dim, hidden_dim, batch_dim, z_dim, drop):
        super(AE, self).__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, z_dim)
            )
        self.dec = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, in_dim)
            )
        self.map = nn.Sequential(
            nn.Linear(batch_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, z_dim),
            )

    def forward(self, x, batch):
        hidden = self.enc(x)
        be = self.map(batch)
        rec_x = self.dec(hidden + be)
#         rec_x = self.dec(hidden)
        return hidden, rec_x

class FBatch(nn.Module):
    """docstring for FBatch"""
    def __init__(self, hidden_dim, batch_dim, z_dim, drop):
        super(FBatch, self).__init__()
        self.clf = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, batch_dim),
            nn.Softmax(dim = -1)
            )

    def forward(self, hidden):
        pred_batch = self.clf(hidden)
        return pred_batch

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''
    def __init__(self, optimizer, lr_lower, lr_upper, n_warmup_steps):
        self._optimizer = optimizer
        self.lr_lower = lr_lower
        self.lr_upper = lr_upper
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps

        if n_steps < n_warmup_steps:
            lr = self.lr_lower + n_steps*(self.lr_upper - self.lr_lower)/n_warmup_steps
        else:
            lr = self.lr_upper - (n_steps - n_warmup_steps)*(self.lr_upper - self.lr_lower)/(2*n_warmup_steps)
            if lr < self.lr_lower:
                lr = self.lr_lower
        return lr

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''
        self.n_steps += 1
        lr = self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr