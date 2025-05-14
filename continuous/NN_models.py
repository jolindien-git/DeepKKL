import torch
from torch import nn
from typing import List


def RK4(f, dt, x, u):
    # -- Runge-Kutta (order 4)
    k1 = f(x, u)
    k2 = f(x + dt / 2 * k1, u)
    k3 = f(x + dt / 2 * k2, u)
    k4 = f(x + dt * k3, u)
    return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def create_mlp(input_dim, output_dim, net_arch, activation_fn = nn.Tanh):
    '''
    returns: a list of nn.Module
    '''
    modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())
    last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
    modules.append(nn.Linear(last_layer_dim, output_dim))
    return nn.Sequential(*modules)


class KKL_Autoencoder(nn.Module):
    
    def __init__(self, x_dim: int, y_dim: int, lambdas: List[float], net_arch: List[int]):
        assert(all(l < 0 for l in lambdas)), "check lambdas values (should be negatives)"
        super().__init__()

        z_dim = len(lambdas)
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.lambdas = lambdas

        self.A = torch.diag(torch.tensor(lambdas, dtype=torch.float))
        self.B = torch.ones((z_dim, y_dim))
        self.T = create_mlp(input_dim=x_dim, output_dim=z_dim, net_arch=net_arch)
        self.invT = create_mlp(input_dim=z_dim, output_dim=x_dim, net_arch=net_arch)

    def z_next(self, z, y, dt):
        z_next = RK4(self.dzdt, dt, z, y)
        return z_next
    
    def dzdt(self, z, y):
        return torch.matmul(self.A, z.unsqueeze(-1)).squeeze(-1) \
            + torch.matmul(self.B, y.unsqueeze(-1)).squeeze(-1)
        
    def encode(self, x):
        return self.T(x)
        
    def decode(self, z):
        return self.invT(z)


def kkl_pde_residuals(x, x_dot, y, model):
    """
    Computes the KKL PDE loss.
    Parameters:
        x: Batch of states x (shape [batch_size, dim_x])
        x_dot: Batch of f(x) (shape [batch_size, dim_x])
        y: Batch of measurements (shape [batch_size, dim_y])
        model: a KKL_Autoencoder
    Returns:
        residuals: shape(batch_size, z_dim)
    """
    z_pred = model.encode(x)
    batch_jacobian = torch.vmap(torch.func.jacrev(model.encode))
    dTdx = batch_jacobian(x)
    residuals = (dTdx @ x_dot.unsqueeze(-1)).squeeze() - model.dzdt(z_pred, y)
    return residuals    


if __name__ == "__main__":
    model = KKL_Autoencoder(x_dim=2,
                            y_dim=1,
                            lambdas=[-.5, -1, -1.5],
                            net_arch=[100, 100])
    
    #-- compute a trajectory of z (for y = 0)
    dt, traj_len = .1, 50
    zs = torch.zeros(traj_len, model.z_dim)
    zs[0] = torch.rand(zs[0].shape)
    for k in range(traj_len - 1):
        y = torch.zeros(model.y_dim)
        zs[k+1] = model.z_next(zs[k], y, dt)
    
    #-- plot
    import matplotlib.pyplot as plt
    plt.plot(torch.arange(traj_len) * dt, zs)
