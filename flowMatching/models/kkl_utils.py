import torch
import torch.nn.functional as F
import numpy as np
from scipy import signal

def get_bessel_dynamics(z_dim, dt, noise_std=0.1, base_bandwidth=5.0, device='cpu', verbose=False):
    '''
    Compute A and B matrices, to obtain the Bessel filter
        z+ = A z + B u      (discrete-time)
    '''
    alpha = 10.0
    cutoff_freq = base_bandwidth / (1.0 + alpha * noise_std)
    _, poles, _ = signal.bessel(N=z_dim, Wn=cutoff_freq, analog=True, output='zpk', norm='phase')
    
    real_poles = poles[np.abs(poles.imag) < 1e-6].real
    complex_poles = poles[np.abs(poles.imag) >= 1e-6]
    if verbose:
        print("Bessel filter, real poles", real_poles)
        print("Bessel filter, complex poles", complex_poles)
    
    A_c = np.zeros((z_dim, z_dim))
    current_idx = 0
    for p in real_poles:
        A_c[current_idx, current_idx] = p
        current_idx += 1
    for p in complex_poles[complex_poles.imag > 0]:
        sigma, omega = p.real, p.imag
        A_c[current_idx, current_idx]     = sigma
        A_c[current_idx, current_idx+1]   = omega
        A_c[current_idx+1, current_idx]   = -omega
        A_c[current_idx+1, current_idx+1] = sigma
        current_idx += 2

    B_c = np.ones((z_dim, 1))

    sys_c = (A_c, B_c, np.eye(z_dim), np.zeros((z_dim, 1)))
    sys_d = signal.cont2discrete(sys_c, dt=dt, method='zoh')
    
    A_d = torch.tensor(sys_d[0], dtype=torch.float32, device=device)
    B_d = torch.tensor(sys_d[1], dtype=torch.float32, device=device)
    return A_d, B_d


class KKL_Latent_Dynamics:
    """
    Generates the KKL observer matrices and simulates the latent state z(t).
    Supports both 'bessel' (Complex FFT) and 'uniform' (Real Conv1D) filters.
    """
    filter_type='bessel'
    
    def __init__(self, y_dim, z_dim, dt=0.01, device='cpu'):
        self.z_dim = z_dim
        self.dt = dt
        self.device = device
        
        if self.filter_type == 'bessel':
            self.A, self.B = get_bessel_dynamics(z_dim, dt, device=device)
            # Diagonalization for FFT method
            self.eigenvals, self.P = torch.linalg.eig(self.A)
            self.P_inv = torch.linalg.inv(self.P)
            
        elif self.filter_type == 'uniform':
            poles = np.linspace(-1, -2.0, z_dim)
            self.A = torch.tensor(np.diag(poles), dtype=torch.float32, device=device)
            self.B = torch.ones((z_dim, y_dim), dtype=torch.float32, device=device)
            
        else:
            raise ValueError("filter_type must be 'bessel' or 'uniform'")

    def compute_z_fast(self, ys_batch):
        """ Routage automatique vers le bon algorithme d'intégration GPU. """
        if self.filter_type == 'bessel':
            return self._compute_z_bessel_fft(ys_batch)
        else:
            return self._compute_z_uniform_conv1d(ys_batch)

    def _compute_z_bessel_fft(self, ys_batch):
        """ Intégration par FFT pour gérer les valeurs propres complexes du filtre de Bessel. """
        B, T_len, _ = ys_batch.shape

        ys_complex = ys_batch.to(dtype=torch.cfloat)
        B_complex = self.B.to(dtype=torch.cfloat)
        
        input_projector = self.P_inv @ B_complex
        u_tilde = torch.einsum('zy, bty -> btz', input_projector, ys_complex)

        range_vec = torch.arange(T_len, device=self.device)
        kernel = self.eigenvals.unsqueeze(0) ** range_vec.unsqueeze(-1)

        n_fft = 2 * T_len
        u_f = torch.fft.fft(u_tilde, n=n_fft, dim=1)
        k_f = torch.fft.fft(kernel, n=n_fft, dim=0)
        
        y_f = u_f * k_f.unsqueeze(0)
        z_tilde_conv = torch.fft.ifft(y_f, n=n_fft, dim=1)[:, :T_len, :]

        z_tilde_shifted = torch.zeros_like(z_tilde_conv)
        z_tilde_shifted[:, 1:, :] = z_tilde_conv[:, :-1, :]

        zs_complex = torch.einsum('zq, btq -> btz', self.P, z_tilde_shifted)
        return zs_complex.real

    def _compute_z_uniform_conv1d(self, ys_batch):
        """ Intégration par Conv1D ultra-rapide pour des pôles strictement réels. """
        B, T_len, _ = ys_batch.shape
        
        v = self.dt * torch.matmul(ys_batch, self.B.T) 
        v_shifted = torch.zeros_like(v)
        v_shifted[:, 1:, :] = v[:, :-1, :]
        v_shifted = v_shifted.permute(0, 2, 1)
        
        a = 1.0 + self.dt * self.A.diagonal() 
        powers = torch.arange(T_len, device=self.device, dtype=torch.float32)
        h = a.unsqueeze(1) ** powers.unsqueeze(0) 
        
        weight = torch.flip(h, dims=[1]).unsqueeze(1) 
        v_padded = F.pad(v_shifted, (T_len - 1, 0))
        
        zs = F.conv1d(v_padded, weight, groups=self.z_dim)
        return zs.permute(0, 2, 1)