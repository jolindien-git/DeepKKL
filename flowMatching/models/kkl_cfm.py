import torch
import torch.nn as nn
import numpy as np


class KKL_CFM(nn.Module):
    """
    Conditional Flow Matching engine for KKL.
    Implements Optimal Transport (OT) flow matching and ODE-based sampling.
    """
    def __init__(self, v_network):
        super().__init__()
        self.v_network = v_network
        self.x_dim = self.v_network.out_proj.out_features

    def compute_loss(self, x1, z):
        """
        Computes the Flow Matching loss using Optimal Transport paths.
        
        Args:
            x1: Ground truth physical states (B, T, x_dim) or (B, M, T, x_dim)
            z: Conditioning latent states (B, T, z_dim) or (B, M, T, z_dim)
        Returns:
            loss: Mean Squared Error of the vector field
        """
        # 1. Sample flow time tau uniformly in [0, 1)
        tau_shape = [x1.shape[0]] + [1] * (x1.dim() - 1)
        tau = torch.rand(tau_shape, device=x1.device) # (B, 1, 1...)
        
        
        # 2. Sample base noise x0 ~ N(0, I)
        x0 = torch.randn_like(x1)

        # 3. Construct the Optimal Transport path (Linear interpolation)
        x_tau = (1 - tau) * x0 + tau * x1

        # 4. The target vector field for OT
        v_target = x1 - x0

        # 5. Predict the vector field using the neural network
        v_pred = self.v_network(tau, x_tau, z)

        # 6. MSE Loss between prediction and Optimal Transport vector
        loss = torch.mean((v_pred - v_target)**2)
        return loss

    import torch

    @torch.no_grad()
    def sample(self, z, n_particles=1, n_steps=20, chunk_size=20):
        """
        Generates physical states x by solving the learned ODE from tau=0 to tau=1.
        Uses chunking along the particle dimension to prevent GPU VRAM bottlenecking.
        
        Args:
            z: Latent condition trajectory (B, T, z_dim)
            n_particles (M): Number of alternative trajectories to generate per batch
            n_steps: Number of integration steps (RK4 method)
            chunk_size: Number of particles to process simultaneously.
            !!! in the case you have memory problems => decrease chunk_size !!!
        Returns:
            x: Generated physical states (B, M, T, x_dim)
        """
        B, T_len, z_dim = z.shape
        device = z.device
        
        # Pre-compute integration steps
        dt = 1.0 / n_steps
        taus = torch.linspace(0, 1.0 - dt, n_steps, device=device)
    
        # 1. Expand z to match the total number of particles M
        # Shape: (B, M, T, z_dim)
        z_expanded = z.unsqueeze(1).expand(B, n_particles, T_len, z_dim)
    
        all_x = []
    
        # 2. Process by chunks along the particle dimension (dim=1)
        for i in range(0, n_particles, chunk_size):
            # Extract current chunk of particles
            # Shape: (B, current_chunk_size, T, z_dim)
            z_chunk = z_expanded[:, i : i + chunk_size]
            current_chunk_size = z_chunk.shape[1]
            
            # Initialize x0 ~ N(0, I) for this chunk
            x = torch.randn(B, current_chunk_size, T_len, self.x_dim, device=device)
    
            # ODE Integration (RK4 method) along the flow time tau
            for tau in taus:
                # k1
                k1 = self.v_network(tau, x, z_chunk)
                
                # k2 (evaluate at tau + dt/2)
                k2 = self.v_network(tau + dt / 2.0, x + (dt / 2.0) * k1, z_chunk)
                
                # k3 (evaluate at tau + dt/2)
                k3 = self.v_network(tau + dt / 2.0, x + (dt / 2.0) * k2, z_chunk)
                
                # k4 (evaluate at tau + dt)
                k4 = self.v_network(tau + dt, x + dt * k3, z_chunk)
                
                # RK4 step update
                x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                
            all_x.append(x)
    
        # 3. Recombine chunks along the particle dimension (dim=1)
        return torch.cat(all_x, dim=1)

    
    def compute_density_map(self,
                            z_cond,
                            x_min, x_max,
                            normalizer,
                            grid_size=80,
                            steps=20,
                            tau_f=1.0):
        """
        Computes the density map p(x|z) using RK4 and 2D divergence.
        Args:
            z_cond: Latent condition (1, z_dim)
            x_min, x_max: bounds of the map
            tau_f: virtual time \in [0, 1]
            steps: Number of integration steps (RK4 method)
        Returns:
            x1: Generated physical states (B, M, T, x_dim)
        """
        device = z_cond.device
        
        # Create the grid
        x_lin1 = np.linspace(x_min[0], x_max[0], grid_size)
        x_lin2 = np.linspace(x_min[1], x_max[1], grid_size)
        X1, X2 = np.meshgrid(x_lin1, x_lin2)
        x_flat = np.vstack([X1.ravel(), X2.ravel()]).T # Shape: (N, 2)
        x = torch.FloatTensor(x_flat).to(device)
        
        # Normalize and prepare condition (N, z_dim)
        x = normalizer.normalize(x)
        N_points = x.shape[0]
        z_batch = z_cond.view(1, -1).expand(N_points, -1)

        def f(tau_scalar, x_in):
            """ Returns velocity (N, 2) and divergence (N,). """
            with torch.enable_grad():
                x_in = x_in.detach().requires_grad_(True)
                tau_batch = torch.ones(x_in.shape[0], 1, device=device) * tau_scalar
                
                v = self.v_network(tau_batch, x_in, z_batch)
                
                # Trace of Jacobian (Divergence)
                grad_v1 = torch.autograd.grad(v[:, 0].sum(), x_in, retain_graph=True)[0]
                grad_v2 = torch.autograd.grad(v[:, 1].sum(), x_in, retain_graph=False)[0]
                
                # Force divergence to be a 1D vector (N,)
                div = (grad_v1[:, 0] + grad_v2[:, 1]).view(-1)
            
            return v.detach(), div.detach()

        # 3. RK4 Integration (tau=1 -> tau=0)
        dtau = tau_f / steps
        current_x = x.clone()
        log_prob_change = torch.zeros(N_points, device=device)
        
        for i in range(steps):
            tau = 1.0 - i * dtau
            
            v1, div1 = f(tau, current_x)
            x2 = current_x - v1 * (dtau / 2)
            v2, div2 = f(tau - dtau/2, x2)
            x3 = current_x - v2 * (dtau / 2)
            v3, div3 = f(tau - dtau/2, x3)
            x4 = current_x - v3 * dtau
            v4, div4 = f(tau - dtau, x4)
            
            # State and log-prob updates
            current_x = current_x - (dtau / 6.0) * (v1 + 2*v2 + 2*v3 + v4)
            current_delta = (dtau / 6.0) * (div1 + 2*div2 + 2*div3 + div4)
            log_prob_change = log_prob_change + current_delta.view(-1)

        # 4. Base distribution log p0 at tau=0
        log_p0 = -0.5 * torch.sum(current_x ** 2, dim=1) - 0.5 * self.x_dim * np.log(2 * np.pi)
        
        # log p1 = log p0 - integral(div)
        log_px = log_p0 - log_prob_change
        
        # Normalizer Jacobian adjustment (sum of log std)
        log_px = log_px - torch.sum(torch.log(normalizer.std.to(device))).item()
        
        density = torch.exp(log_px).reshape(grid_size, grid_size).cpu().numpy()
        return X1, X2, density