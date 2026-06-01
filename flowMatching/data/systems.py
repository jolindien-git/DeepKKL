"""
Dynamic systems dataset generation and definitions.
Compatible with both NumPy arrays (for dataset generation) 
and PyTorch tensors (for vectorized observer evaluation).
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import copy


def RK4(f, dt, x, u=None):
    """Agnostic Runge-Kutta 4 integrator (works with NumPy and PyTorch)."""
    k1 = f(x, u)
    k2 = f(x + dt / 2 * k1, u)
    k3 = f(x + dt / 2 * k2, u)
    k4 = f(x + dt * k3, u)
    return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


class KKL_Dataset(Dataset):
    """
    Base class for system datasets.
    Generates trajectories and provides dynamics/observation functions.
    """
    dt = 0 
    x_dim = -1 
    y_dim = -1 
    u_dim = -1 
    x0_high = np.float32([0] * x_dim) 
    x0_low = -x0_high
    n_modes = 1 # undistinguishable modes
    name = "KKL_Dataset"
    
    def __init__(self, n_trajs: int, traj_len: int=1, noise_std: float=.0, process_std: float=.0):
        self.n_trajs = n_trajs
        self.traj_len = traj_len
        self.noise_std = noise_std
        ts, xs, ys, _ = self.generate_trajectories(
            n_trajs, traj_len, autonomous=True, noise_std=noise_std, process_std=process_std
        )
        self.ts, self.xs, self.ys = ts[..., None], xs, ys

    def __len__(self):
        return self.n_trajs

    def __getitem__(self, index):
        return self.ts[index], self.xs[index], self.ys[index]
        
    @staticmethod
    def get_derivs(x, u=None):
        raise NotImplementedError
        
    def get_x_next(self, x, u=None, process_std=.0):
        def get_derivs_wrapper(x_val, u_val):
            return self.get_derivs(x_val, u_val)
        
        x_next = RK4(get_derivs_wrapper, self.dt, x, u)
        if process_std > 0:
            x_next += self.dt * np.float32(np.random.normal(0, process_std, size=x.shape))
        return x_next
    
    @staticmethod
    def get_y(x):
        raise NotImplementedError
    
    def get_u(self, t, x=None):
        raise NotImplementedError
        
    def get_true_modes(self, x_traj):
        """
        Returns all valid topological branches for a given true trajectory.
        Defaults to C=1 (only the true trajectory itself).
        Override in subclasses if the system has symmetries/indistinguishability.
        
        Args:
            x_traj: Tensor or Array of shape (..., T, D)
        Returns:
            modes: Tensor or Array of shape (..., C, T, D)
        """
        if isinstance(x_traj, torch.Tensor):
            return x_traj.unsqueeze(-3)
        return np.expand_dims(x_traj, axis=-3)

    def generate_trajectories(self, n_traj, traj_len, autonomous=True, noise_std=.0, process_std=.0):
        ts = np.zeros((n_traj, traj_len), dtype=np.float32)
        xs = np.zeros((n_traj, traj_len, self.x_dim), dtype=np.float32)
        us = np.zeros((n_traj, traj_len, self.u_dim), dtype=np.float32)
        
        xs[:, 0, :] = np.random.uniform(self.x0_low, self.x0_high, xs[:, 0, :].shape)
        
        t = 0
        for k in range(traj_len - 1):
            x = xs[:, k, :]
            if autonomous:
                xs[:, k + 1, :] = self.get_x_next(x, process_std=process_std)
            else:
                us[:, k, 0] = self.get_u(t, x)
                xs[:, k + 1, :] = self.get_x_next(x, us[:, k, :], process_std=process_std)            
            t += self.dt
            ts[:, k+1] = t
            
        ys = copy.deepcopy(self.get_y(xs))
        if noise_std > 0:
            ys += np.float32(np.random.normal(0, noise_std, size=ys.shape))

        return ts, xs, ys, us    


class OscillatorWithParameter(KKL_Dataset):
    x_dim, y_dim, u_dim = 3, 1, 0
    x0_high = np.float32([1.5, 1.5, 1.2])
    x0_low = np.float32([-1.5, -1.5, .8])
    dt = .01
    name = "Oscillator_With_Parameter"
    
    @staticmethod
    def get_derivs(x, u=None):
        dxdt = 0 * x
        x1, x2, x3 = x[..., 0], x[..., 1], x[..., 2]
        dxdt[..., 0] = x2
        dxdt[..., 1] = -x1 * x3
        dxdt[..., 2] = 0
        return dxdt
    
    @staticmethod
    def get_y(x):
        return x[..., :1]
    

class VanDerPol(KKL_Dataset):
    x_dim, y_dim, u_dim = 2, 1, 0
    x0_high = np.float32([2, 2])
    x0_low = -x0_high
    dt = .05
    name = "Van der Pol"

    @staticmethod
    def get_derivs(x, u=None):
        dxdt = 0 * x
        x1, x2 = x[..., 0], x[..., 1]
        dxdt[..., 0] = x2
        dxdt[..., 1] = (1 - x1**2) * x2 - x1
        return dxdt
    
    @staticmethod
    def get_y(x):
        return x[..., :1] 
    

class DoublePendulum(KKL_Dataset):
    x_dim, u_dim = 4, 0
    y_dim = 1
    x0_high = np.float32([np.pi, np.pi, 2.0, 2.0])
    x0_low = -x0_high
    dt = 0.01
    name = "Double Pendulum"

    # --- Physical Parameters
    m1 = 1. #2.
    m2 = 1. #2.
    l1 = 1. #1.5
    l2 = 1. #1.5
    g = 9.81
    
    @staticmethod
    def get_derivs(x, u=None):
        dxdt = 0 * x
        th1, th2 = x[..., 0], x[..., 1]
        dth1, dth2 = x[..., 2], x[..., 3]
        
        m1 = DoublePendulum.m1
        m2 = DoublePendulum.m2
        l1 = DoublePendulum.l1
        l2 = DoublePendulum.l2
        g = DoublePendulum.g
        
        delta = th1 - th2
        
        den_base = 2 * m1 + m2 - m2 * np.cos(2 * delta)
        den1 = l1 * den_base
        den2 = l2 * den_base
        
        dxdt[..., 0] = dth1
        dxdt[..., 1] = dth2
        
        num1 = (-g * (2 * m1 + m2) * np.sin(th1) 
                - m2 * g * np.sin(th1 - 2 * th2) 
                - 2 * np.sin(delta) * m2 * (dth2**2 * l2 + dth1**2 * l1 * np.cos(delta)))
        dxdt[..., 2] = num1 / den1
        
        num2 = (2 * np.sin(delta) * (dth1**2 * l1 * (m1 + m2) 
                + g * (m1 + m2) * np.cos(th1) 
                + dth2**2 * l2 * m2 * np.cos(delta)))
        dxdt[..., 3] = num2 / den2
        
        return dxdt
    
    @staticmethod
    def get_y(x):
        # th1 = x[..., 0]
        # return np.stack([np.cos(th1), np.sin(th1)], axis=-1)
        return x[..., :1] 


class Rossler(KKL_Dataset):
    x_dim, y_dim, u_dim = 3, 1, 0
    x0_high = np.float32([1] * 3)
    x0_low = -x0_high
    dt = .05
    name = "Rossler"
    
    @staticmethod
    def get_derivs(x, u=None):
        dxdt = 0 * x
        x1, x2, x3 = x[..., 0], x[..., 1], x[..., 2]
        dxdt[..., 0] = -x2 - x3
        dxdt[..., 1] = x1 + .2 * x2
        dxdt[..., 2] = .2 + x3 * (x1 - 5.7)
        return dxdt
    
    @staticmethod
    def get_y(x):
        return x[..., 1:2] 


class LinearDyn_PolynomOut(KKL_Dataset):
    x_dim, y_dim, u_dim = 2, 1, 0
    x0_high = np.float32([.5, .5])
    x0_low = -x0_high
    dt = .01
    name = "Linear Dynamics With Nonlinear Output"
    
    @staticmethod
    def get_derivs(x, u=None):
        dxdt = 0 * x
        x1, x2 = x[..., 0], x[..., 1]
        dxdt[..., 0] = x2
        dxdt[..., 1] = -x1
        return dxdt
    
    @staticmethod
    def get_y(x):
        x1, x2 = x[..., :1], x[..., 1:]
        return x1**2 - x2**2 + x1 + x2


class LotkaVolterra(KKL_Dataset):
    x_dim, y_dim, u_dim = 2, 1, 0
    x0_high = np.float32([1] * 2) * 6
    x0_low = x0_high * 0
    dt = 0.02
    name = "Lotka Volterra"
    
    @staticmethod
    def get_derivs(x, u=None):
        c1, c2 = 3., 2.
        dxdt = 0 * x
        x1, x2 = x[..., 0], x[..., 1]
        dxdt[..., 0] = x1 - x1 * x2 / c2
        dxdt[..., 1] = -x2 + x2 * x1 / c1
        return dxdt
    
    @staticmethod
    def get_y(x):
        return x[..., 0:1] + x[..., 1:2] 
    
    def get_x_next(self, x, u=None, process_std=.0):
        x_next = super().get_x_next(x, u, process_std)
        lib = torch if isinstance(x, torch.Tensor) else np
        return lib.where(x_next > 1e-4, x_next, 1e-4)


class Duffing_Indistinguishable(KKL_Dataset):
    x_dim, y_dim, u_dim = 2, 1, 0
    x0_high = np.float32([2, 2])
    x0_low = -x0_high
    dt = 0.02
    name = "Duffing Indistinguishable"
    
    @staticmethod
    def get_derivs(x, u=None):
        dxdt = 0 * x
        x1, x2 = x[..., 0], x[..., 1]
        dxdt[..., 0] = x2
        dxdt[..., 1] = x1 - x1**3
        return dxdt
    
    @staticmethod
    def get_y(x):
        x1, x2 = x[..., 0:1], x[..., 1:2]
        return -x1**2 / 2 + x2**2 / 2 + x1**4 / 4 
        


class BiModal(KKL_Dataset):
    x_dim, y_dim, u_dim = 2, 2, 0
    x0_high = np.float32([2, 2])
    x0_low = -x0_high
    dt = .05
    name = "BiModal"
    n_modes = 2 # undistinguishable modes
    
    @staticmethod
    def get_derivs(x, u=None):
        dxdt = 0 * x
        x1, x2 = x[..., 0], x[..., 1]
        dxdt[..., 0] = x2 + x1 * (1 - (x1**2 + x2**2))
        dxdt[..., 1] = -x1 + x2 * (1 - (x1**2 + x2**2))
        return dxdt
    
    @staticmethod
    def get_y(x):
        y = 0 * x
        y[..., 0] = x[..., 0]**2 - x[..., 1]**2
        y[..., 1] = 2 * x[..., 0] * x[..., 1]
        return y

    def get_true_modes(self, x_traj):
        """
        Symmetry: x and -x yield the exact same output y.
        """
        if isinstance(x_traj, torch.Tensor):
            return torch.stack([x_traj, -x_traj], dim=-3)
        return np.stack([x_traj, -x_traj], axis=-3)


class QuadModeCoupledVDP(KKL_Dataset):
    """
    4D System: Two coupled Van der Pol oscillators.
    States: x1 (pos1), x2 (vel1), x3 (pos2), x4 (vel2)
    Observation: y = [x1^2, x3^2] (y_dim = 2).
    Creates 4 combinatorial modes: (+x1,+x3), (+x1,-x3), (-x1,+x3), (-x1,-x3)
    """
    x_dim = 4
    y_dim = 2
    u_dim = 0
    n_modes = 4
    dt = 0.05
    x0_low = np.array([-2.0, -2.0, -2.0, -2.0])
    x0_high = np.array([2.0, 2.0, 2.0, 2.0])
    mu = 1.0      
    k = 0.1       # Lower coupling to prevent phase locking
    w1 = 1.0      # Natural frequency mass 1
    w2 = 1.414    # Natural frequency mass 2 (irrational ratio w2/w1)

    def get_derivs(self, x, u=None):
        """ Computes the time derivative with asymmetric frequencies. """
        dxdt = 0 * x 
        x1, x2, x3, x4 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
        dxdt[..., 0] = x2
        dxdt[..., 1] = self.mu * (1 - x1**2) * x2 - (self.w1**2) * x1 + self.k * (x3 - x1)
        dxdt[..., 2] = x4
        dxdt[..., 3] = self.mu * (1 - x3**2) * x4 - (self.w2**2) * x3 + self.k * (x1 - x3)
        
        return dxdt

    def get_y(self, x):
        """ 
        Observation: Squared position of BOTH masses.
        This provides observability but creates 4 distinct modes.
        """
        if isinstance(x, torch.Tensor):
            return torch.stack([x[..., 0]**2, x[..., 2]**2], dim=-1)
        else:
            return np.stack([x[..., 0]**2, x[..., 2]**2], axis=-1)

    def get_true_modes(self, xs_true):
        """
        Combinatorial expansion of the 4 indistinguishable true modes.
        """
        m1 = xs_true
        m2 = xs_true * 1;  m2[..., 0:2] *= -1
        m3 = xs_true * 1;  m3[..., 2:4] *= -1
        m4 = xs_true * -1
        
        xp = torch if hasattr(xs_true, 'device') else np
        if xp is torch:
            return torch.stack([m1, m2, m3, m4], dim=1)
        else:
            return np.stack([m1, m2, m3, m4], axis=1)
        

datasets = {
    'VDP': VanDerPol,
    'Param': OscillatorWithParameter,
    'Rossler': Rossler,
    'LV': LotkaVolterra,
    'CDC19': LinearDyn_PolynomOut,
    'Duffing': Duffing_Indistinguishable,
    'BiModal': BiModal,
    'VDP2': QuadModeCoupledVDP,
    'Double' : DoublePendulum,
}