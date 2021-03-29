"""
Created: March 2021

@author: johan Peralez
"""

import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class VDP_Dataset(Dataset):
    """
    data:
        x: shape = (n_samples, 2)
        y: shape = (n_samples, 1)
    """
    x_dim, y_dim, u_dim = 2, 1, 1
    dt = .01

    def __init__(self, n_samples, std_x=None):
        super().__init__()

        # init x at random values
        x0_high = np.array([2.3, 3.3])
        x = np.float32(np.random.uniform(-x0_high, x0_high, (n_samples, self.x_dim)))
        x_next = self.get_x_next(x)
        y = np.expand_dims(x[:, 0], -1)
        y_next = np.expand_dims(x_next[:, 0], -1)

        if std_x is None:
            std_x = x.std()

        self.x = x
        self.x_next = x_next
        self.y = y
        self.y_next = y_next
        self.n_samples = n_samples
        self.std_x = std_x

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        x = self.x[index, :] / self.std_x
        y = self.y[index, :]
        x_next = self.x_next[index, :] / self.std_x
        y_next = self.y_next[index, :]
        return x, y, x_next, y_next

    def get_x_next(self, x, u=None):
        x_next = x + self.dt * self.fn_derivs(x, u)
        return x_next

    def fn_derivs(self, x, u=None):
        """
        required x shape: (brodcast_dim, x_dim)
        """
        dxdt = np.zeros_like(x)
        x1, x2 = x[:, 0], x[:, 1]
        dxdt[:, 0] = x2
        dxdt[:, 1] = (1 - x1**2) * x2 - x1
        if u is not None:
            dxdt[:, 1] += u[:, 0]
        return dxdt

    def generate_sequences(self, n_simus, simus_len, autonomous=True, noise_std=0):
        x0_high = np.array([2.3, 3.3])

        x = np.zeros((n_simus, simus_len, self.x_dim), dtype=np.float32)
        u = np.zeros((n_simus, simus_len, self.u_dim), dtype=np.float32)

        # init x at random values
        x[:, 0, :] = np.random.uniform(-x0_high, x0_high, x[:, 0, :].shape)

        # time sequences
        t = 0
        for k in range(simus_len - 1):
            u[:, k, 0] = 0 if autonomous else .44 * np.cos(.5 * t)
            x[:, k + 1, :] = self.get_x_next(x[:, k, :], u[:, k, :])

            t += self.dt

        y = np.expand_dims(x[:, :, 0], -1)
        y = y + np.float32(np.random.normal(0, noise_std, size=y.shape))

        if self.std_x is not None:
            x = x / self.std_x
        return x, y, u

    def render(self):
        n_points = min(200, self.n_samples)

        x = self.x[: n_points, :]
        x_next = self.x_next[: n_points, :]

        plt.figure()
        plt.style.use("seaborn")

        # plot x
        plt.plot(x[:, 0], x[:, 1], '.')
        plt.plot(x_next[:, 0], x_next[:, 1], '.')
        plt.legend(['$x_k$', '$x_{k+1}$'])
