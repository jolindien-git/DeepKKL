"""
Created: November 2023
Modified:
@author: Johan Peralez
"""

import numpy as np
import torch
from torch.optim import lr_scheduler
from dataclasses import dataclass, field
from torch.utils.data import random_split

from dataset import KKL_Dataset


@dataclass
class Problem:
    dataset: KKL_Dataset
    noise_std: float  # Gaussian noise std: N(0, std)
    data_traj_number: int  # training data
    data_traj_len: int  # train trajectories length (time-steps)
    name: str


@dataclass
class Algo:
    # -- KKL
    A_diag: bool = False
    z_dim: int = 5
    # -- neural network training
    criterion: type(torch.nn.MSELoss()) = torch.nn.HuberLoss(delta=.1)
    batch_size: int = 100
    net_arch: list = field(default_factory=list)  # hidden layers size
    epochs: int = 20
    lr_init: float = 1e-3


def get_data(dataset,
             traj_number,
             traj_len,
             noise_std,
             split_percent: float = None):
    """
    Parameters
        dataset: class of the dataset (e.g. VanDerPol_Dataset)
        split_percent: for splitting the data between train / valid
    Returns
        One (or two) pytorch dataset corresponding to the problem
        If split_percent is not None than return two datasets
    """
    data = dataset(use_traj=True,
                   n_samples=traj_number,
                   traj_len=traj_len,
                   noise_std=noise_std)
    if split_percent is None:
        return data
    train_size = len(data) * split_percent // 100
    valid_size = len(data) - train_size
    train_data, valid_data = random_split(data, [train_size, valid_size])
    return train_data, valid_data


def get_problem_data(pb: Problem, split_percent: float = None):
    return get_data(pb.dataset, pb.data_traj_number, pb.data_traj_len,
                    pb.noise_std, split_percent)


def train_autoencoder(
        model,  # the neural network (modified inplace)
        train_loader,
        valid_loader,
        algo
        ):
    print("==== training model ====")

    def get_loss(xs, ys):
        if model.use_encoder:
            zs_dyn, xs_decoder = model.trajectories(ys, xs[:, 0, :])
            zs = model.encode(xs)
            loss = algo.criterion(xs_decoder, xs) \
                + 1e-2 * algo.criterion(zs, zs_dyn)
        else:
            _, xs_decoder = model.trajectories(ys)
            loss = algo.criterion(xs_decoder, xs)
        return loss
    
    def eval():
        with torch.no_grad():
            for i, (xs, ys) in enumerate(valid_loader):
                assert(i == 0)
                loss = get_loss(xs, ys)
        return loss.item()
        
    #-- main loop
    optimizer = torch.optim.Adam(model.parameters(), lr=algo.lr_init)
    optimizer2 = torch.optim.Adam(model.encoder.parameters(), lr=algo.lr_init)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=algo.epochs)
    scheduler2 = lr_scheduler.CosineAnnealingLR(optimizer2, T_max=algo.epochs)
    best_loss = np.inf
    for epoch in range(algo.epochs):
        if epoch == 0:
            #-- first evaluation
            print('epoch %i valid %.2e' % (-1, eval()))
        train_losses = []
        #-- train
        for xs, ys in train_loader:
            if model.use_encoder:
                zs_dyn, xs_decoder = model.trajectories(ys, xs[:, 0, :])
                zs = model.encode(xs)
                loss = algo.criterion(xs_decoder, xs)
                loss2 = algo.criterion(zs, zs_dyn)
                
                loss.backward(retain_graph=True)
                loss2.backward()
                
                optimizer.step()
                train_losses.append(loss.item())
                optimizer.zero_grad()
                
                
                optimizer2.step()
                optimizer2.zero_grad()
                
            else:
                loss = get_loss(xs, ys)    
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
                optimizer.zero_grad()
        #-- verbose: learning rate, train loss, valid loss
        lr = scheduler.get_last_lr()[0]
        valid_loss = eval()
        print('epoch %i train %.2e    valid %.2e    lr %.2e' %
                  (epoch, np.mean(train_losses), valid_loss, lr))
        #-- save best model
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model, "models/train_decoder.pth")
        #-- learning rate
        scheduler.step()
        scheduler2.step()

    model = torch.load("models/train_decoder.pth")