import random
import torch
import numpy as np


def set_seed(seed):
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        

class Normalizer:
    """ Normalizes and unnormalizes physical states. """
    def __init__(self, xs_train):
        if not isinstance(xs_train, torch.Tensor):
            xs_train = torch.tensor(xs_train, dtype=torch.float32)
            
        # Flatten to (Total_Points, x_dim) to get 1D stats of shape (x_dim,)
        flat_x = xs_train.reshape(-1, xs_train.shape[-1])
        self.mean = flat_x.mean(dim=0)
        self.std = flat_x.std(dim=0)
        self.std[self.std < 1e-8] = 1.0 

    def normalize(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def unnormalize(self, x_norm):
        return x_norm * self.std.to(x_norm.device) + self.mean.to(x_norm.device)