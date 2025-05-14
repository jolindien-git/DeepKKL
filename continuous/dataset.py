import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import copy


# -- Runge-Kutta (order 4)
def RK4(f, dt, x, u):
    k1 = f(x, u)
    k2 = f(x + dt / 2 * k1, u)
    k3 = f(x + dt / 2 * k2, u)
    k4 = f(x + dt * k3, u)
    return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


class KKL_Dataset(Dataset):
    """
    if trajectories
        data (numpy array)
            x: shape (n_traj, traj_len, x_dim)
            y: shape (n_traj, traj_len, y_dim)
        __getitem__
            xs, ys
    else
        data (numpy array)
            x, x_next: shape = (n_samples, x_dim)
            y, y_next: shape = (n_samples, y_dim)
        __getitem__
            x, y, x_next, y_next
    """
    dt = 0 # step time
    x_dim = -1 # state dimension
    y_dim = -1 # output (measurement) dimension
    u_dim = -1 # control dimension
    x_high = np.float32([0] * x_dim) # for random values
    x0_high = np.float32([0] * x_dim) # for initial rnd values (trajectories)
    x_low = -x_high
    x0_low = -x0_high
    name = "KKL_Dataset"
    
    def __init__(self,
                 n_samples: int, # dataset size
                 use_traj: bool, # generates trajectory or couple of points
                 traj_len: int=1, # trajectories length
                 noise_std: float=0
                 ):
        self.use_traj = use_traj
        self.n_samples = n_samples
        if use_traj:
            ts, xs, ys, _ = self.generate_trajectories(n_samples, traj_len,
                                                       autonomous=True,
                                                       noise_std=noise_std)
            self.ts, self.xs, self.ys = ts, xs, ys
        else:
            #-- init x at random values
            self.x = np.float32(np.random.uniform(self.x_low, self.x_high,
                                                  (n_samples, self.x_dim)))
            #-- compute (x_next, y, y_next)
            self.x_dot = self.get_derivs(self.x)
            self.x_next = self.get_x_next(self.x)
            self.y = copy.deepcopy(self.get_y(self.x))
            self.y_next = copy.deepcopy(self.get_y(self.x_next))
            if noise_std > 0:
                self.y += np.float32(np.random.normal(0, noise_std,
                                                      size=self.y.shape))
                self.y_next += np.float32(np.random.normal(0, noise_std,
                                                      size=self.y_next.shape))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        if self.use_traj:
            return (self.xs[index],
                    self.ys[index])
        else:
            return (self.x[index], self.x_dot[index],
                    self.y[index])
    
    def get_derivs(self, x, u=None):
        raise NotImplementedError
        
    def get_x_next(self, x, u=None):
        '''
        Parameters
            x: shape = (brodcast_dim, x_dim)
            u: shape = (brodcast_dim, u_dim)
        Returns
            x_next: shape = (brodcast_dim, x_dim)
        '''
        # x_next = x + self.dt * self.get_derivs(x, u)
        x_next = RK4(self.get_derivs, self.dt, x, u)
        return x_next
    
    def get_y(self, x):
        '''
        Get the outputs (measurements) corresponding to the states x.
        Parameters
            x: shape = (brodcast_dim, x_dim)
        Returns
            y: shape = (brodcast_dim, y_dim)
        '''
        raise NotImplementedError
    
    def get_ys(self, xs):
        '''
        Get outputs (measurements) corresponding to the states x.
        Parameters
            xs: shape = (n_traj, traj_len, x_dim)
        Returns
            ys: shape = (n_traj, traj_len, y_dim)
        '''
        raise NotImplementedError
    
    def get_u(self, t, x=None):
        '''
        Controller.
        Parameters
            t: float
            x: shape = (brodcast_dim, x_dim)
        Returns
            u: shape = (brodcast_dim, u_dim) or None
        '''
        raise NotImplementedError
        
    def generate_trajectories(self, n_traj, traj_len,
                              autonomous=True, noise_std=0):
        """
        returns t, x, y, u where
            t shape: (n_traj, traj_len)
            x shape: (n_traj, traj_len, x_dim)
            y shape: (n_traj, traj_len, y_dim)
            u shape: (n_traj, traj_len, u_dim)
        """
        ts = np.zeros((n_traj, traj_len), dtype=np.float32)
        xs = np.zeros((n_traj, traj_len, self.x_dim), dtype=np.float32)
        us = np.zeros((n_traj, traj_len, self.u_dim), dtype=np.float32)
        #-- init x at random values
        xs[:, 0, :] = np.random.uniform(self.x0_low,
                                        self.x0_high,
                                        xs[:, 0, :].shape)
        #-- make time-steps
        t = 0
        for k in range(traj_len - 1):
            x = xs[:, k, :]
            if autonomous:
                xs[:, k + 1, :] = self.get_x_next(x)
            else:
                us[:, k, 0] = self.get_u(t, x)
                xs[:, k + 1, :] = self.get_x_next(x, us[:, k, :])            
            t += self.dt
            ts[:, k+1] = t
        ys = copy.deepcopy(self.get_ys(xs))
        if noise_std > 0:
            ys += np.float32(np.random.normal(0, noise_std, size=ys.shape))

        return ts, xs, ys, us
    
    def render(self):
        if self.use_traj:
            n_traj = 3
            traj_idx = np.random.choice(list(range(len(self))), n_traj)
            ts = self.ts[traj_idx, :]
            xs = self.xs[traj_idx, :, :]
            ys = self.ys[traj_idx, :, :]
            
            #-- plot a trajectory
            plt.figure()
            n_subplots = self.x_dim + self.y_dim
            for i in range(self.x_dim):
                #-- plot states
                plt.subplot(n_subplots, 1, i + 1)
                for traj in range(n_traj):
                    plt.plot(ts[traj], xs[traj, :, i])
                plt.legend(['$x_%i$' % (i + 1,)] * n_traj)
                if i == 0:
                    plt.title(self.name)
            for i in range(self.y_dim):
                #-- plot outputs
                plt.subplot(n_subplots, 1, self.x_dim + i + 1)
                for traj in range(n_traj):
                    plt.plot(ts[traj], ys[traj, :, i])
                plt.legend(['$y_%i$' % (i + 1,)] * n_traj)
            plt.xlabel('t')
        else:
            n_points = min(500, self.n_samples)
            x = self.x[: n_points, :2] # couples (x1, x2)
            x_next = self.x_next[: n_points, :2] # couples (x1_next, x2_next)
    
            #-- plot points in the phase plane (x1, x2)
            plt.figure()
            plt.plot(x[:, 0], x[:, 1], '.')
            plt.plot(x_next[:, 0], x_next[:, 1], '.')
            plt.legend(['$x_k$', '$x_{k+1}$'])
            plt.title(self.name)
            plt.xlabel('$x_1$')
            plt.ylabel('$x_2$')


class Example_Dataset(KKL_Dataset):
    """
    System:
        \dot x_1 = x_2
        \dot x_2 = -x_1 * x_3 
        \dot x_3 = 0
        y = x_1
    """
    x_dim, y_dim, u_dim = 3, 1, 0
    x0_high = np.float32([1.5, 1.5, 1.2])
    x0_low = np.float32([-1.5, -1.5, .8])
    x_high, x_low = x0_high, x0_low
    dt = .01
    name = "Example"
    
    def __init__(self, n_samples, use_traj, traj_len=1, noise_std=0):
        super().__init__(n_samples, use_traj, traj_len, noise_std)
        
    def get_derivs(self, x, u=None):
        dxdt = np.zeros_like(x)
        x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
        dxdt[:, 0] = x2
        dxdt[:, 1] = -x1 * x3
        dxdt[:, 2] = 0
        return dxdt
    
    def get_y(self, x):
        return x[..., :1]
    
    def get_ys(self, xs):
        return self.get_y(xs)
    
    def get_u(self, t, x=None):
        return 0


if __name__ == "__main__":
    dataset = Example_Dataset(n_samples=40, use_traj=True, traj_len=1000)
    dataset.render()
