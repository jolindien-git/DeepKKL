"""
Created: March 2021
Modified: November 2023
@author: johan Peralez
"""

import torch
from torch import nn
from scipy import linalg


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


class KKL_AutoEncoder(nn.Module):
    
    def __init__(self,
                 x_dim,
                 y_dim,
                 z_dim,
                 net_arch,
                 A_diag=False,
                 use_encoder=False):
        super().__init__()
        self.z_dim = z_dim
        self.A_diag = A_diag # matrix A is diagonal (True/False)
        self.use_encoder = use_encoder
        #-- A is learnable (initialized as a diagonal matrix)
        diag_ini = torch.linspace(.85, .95, z_dim)
        if A_diag:
            #-- A is constrained to be diagonal
            self.diag = nn.Parameter(diag_ini)
            self.A = torch.diag(self.diag)
        else:
            #-- A is full
            self.A = nn.Linear(z_dim, z_dim, bias=False)
            self.A.weight.data = torch.diag(diag_ini)
        self.A_frozen = False
        #-- B is fixed
        self.B = torch.ones((z_dim, y_dim)) #* dt
        #-- decoder = MLP
        self.decoder = create_mlp(z_dim, x_dim, net_arch)
        #-- encoder
        self.encoder = create_mlp(x_dim, z_dim, net_arch)
        self._encoder = create_mlp(y_dim, z_dim, [v // 2 for v in net_arch])
    
    def z_next(self, z, y):
        #-- apply the dynamics: z_next = A z + B y
        if self.A_diag:
            self.A = torch.diag(self.diag)
            return (self.A @ z.unsqueeze(-1)).squeeze(-1) \
                + (self.B @ y.unsqueeze(-1)).squeeze(-1)
        else:
            return self.A(z) + (self.B @ y.unsqueeze(-1)).squeeze(-1)
    
    def decode(self, z):
        return self.decoder(z)
    
    def encode(self, x):
        return self.encoder(x)
    
    def _encode(self, y):
        return self._encoder(y)
    
    def forward(self, z):
        return self.decode(z)
    
    def trajectories(self, ys, xs0=None):
        """
        parameters:
            ys: tensor with shape(batch_size, traj_len, y_dim)
        returns:
            zs_dyn: tensor with shape(batch_size, traj_len, z_dim)
            xs_decoder: tensor with shape(batch_size, traj_len, x_dim)
            xs0: tensor with shape(batch_size, x_dim)
        """
        batch_size, traj_len, y_dim = ys.shape
        #-- z_dyn trajectories
        if xs0 is None:
            #-- initializes at 0, i.e. z0 = 0
            # z_dyn = torch.zeros(batch_size, self.z_dim)
            z_dyn = self._encode(ys[:, 0, :])
        else:
            #-- initializes with the encoder, i.e. z0 = T(x0)
            z_dyn = self.encode(xs0)
        zs_dyn = [z_dyn]
        for k in range(traj_len - 1):
            if self.A_frozen:
                with torch.no_grad():
                    z_dyn = self.z_next(z_dyn, ys[:, k, :])
            else:
                z_dyn = self.z_next(z_dyn, ys[:, k, :])
            zs_dyn.append(z_dyn)
        zs_dyn = torch.stack(zs_dyn, dim=1)
        #-- decode
        xs_decoder = self(zs_dyn)
        return zs_dyn, xs_decoder


#%% utils

def render_eigenvalues(A):
    '''
    Parameters
        M: a numpy matrix
    '''
    try:
        A = A.clone().detach().numpy()
    except:
        A = A.weight.data.numpy()
    eigs = linalg.eig(A, right=False)
    #-- print the eigenvalues
    print('\t eig(A) [%s]' % 
          (", ".join('{0.real:.3f} + {0.imag:.3f}i'.format(v) for v in eigs),))
    #-- print their norm
    print("\t |eig(A)| [%s]" % 
          (", ".join('{0.real:.3f}'.format(linalg.norm(v)) for v in eigs),))


def render_layers_norm(mlp):
    for layer in mlp:
        if type(layer) == torch.nn.modules.linear.Linear:
            print(torch.linalg.matrix_norm(layer.weight.data, ord=2))
            # print(linalg.svd(layer.weight.data.detach().numpy(), compute_uv=False))
            