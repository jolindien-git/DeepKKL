import torch
from torch import nn
from typing import List, Type


def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.Tanh
) -> List[nn.Module]:

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())
        
    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    
    return nn.Sequential(*modules)


class KKL_Autoencoder(nn.Module):
    
    def __init__(self, x_dim: int, y_dim: int, dt: float, lambdas: List[float], net_arch: List[int]):
        assert(all(l < 0 for l in lambdas))
        super().__init__()

        z_dim = len(lambdas)
        self.dt = dt
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.lambdas = lambdas

        self.A = torch.diag(1 + dt * torch.tensor(lambdas, dtype=torch.float))
        self.B = dt * torch.ones((z_dim, y_dim))
        self.T = create_mlp(input_dim=x_dim, output_dim=z_dim, net_arch=net_arch)
        self.Psi = create_mlp(input_dim=z_dim, output_dim=x_dim, net_arch=net_arch)

    def z_next(self, z, y):
        return torch.matmul(self.A, z.unsqueeze(-1)).squeeze(-1) + torch.matmul(self.B, y.unsqueeze(-1)).squeeze(-1)
        
    def encode(self, x):
        return self.T(x)
        
    def decode(self, z):
        return self.Psi(z)    
