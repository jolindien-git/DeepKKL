import torch
from models.observer import BaseMultimodalObserver
from data.systems import RK4
from utils.tracking import track_modes

class ParticleFilterObserver(BaseMultimodalObserver):
    """
    Vectorized Bootstrap Particle Filter in PyTorch.
    """
    def __init__(self, dataset, n_particles=2000, n_modes=2, 
                 noise_std=1.0, process_std=0.0, device='cpu'):
        super().__init__(dataset.x_dim, dataset.y_dim, dataset.dt, device)
        self.n_particles = n_particles
        self.n_modes = n_modes
        self.noise_std = noise_std if noise_std > 1e-8 else 1e-2
        self.process_std = process_std
        
        self.x0_low = torch.tensor(dataset.x0_low, device=device)
        self.x0_high = torch.tensor(dataset.x0_high, device=device)
        self.get_derivs = dataset.get_derivs
        self.get_y = dataset.get_y

    def forward(self, ys_batch):
        B, T_len, _ = ys_batch.shape
        
        particles = torch.empty((B, self.n_particles, self.x_dim), device=self.device)
        for d in range(self.x_dim):
            particles[..., d].uniform_(self.x0_low[d].item(), self.x0_high[d].item())
            
        weights = torch.ones((B, self.n_particles), device=self.device) / self.n_particles
        
        p_hist = torch.zeros((B, T_len, self.n_particles, self.x_dim), device=self.device)
        w_hist = torch.zeros((B, T_len, self.n_particles), device=self.device)

        for t in range(T_len):
            y_t = ys_batch[:, t, :]
            
            # 1. Predict
            if t > 0:
                particles = RK4(self.get_derivs, self.dt, particles)
                if self.process_std > 1e-8:
                    noise = torch.randn_like(particles) * self.process_std * self.dt
                    particles += noise
            
            # 2. Update
            y_pred = self.get_y(particles)
            diff = y_pred - y_t.unsqueeze(1)
            
            log_w = -0.5 * torch.sum(diff**2, dim=-1) / (self.noise_std**2)
            log_w = log_w - log_w.max(dim=1, keepdim=True).values 
            weights = torch.exp(log_w)
            weights = weights / weights.sum(dim=1, keepdim=True)
            
            p_hist[:, t] = particles
            w_hist[:, t] = weights
            
            # 3. Resample
            neff = 1.0 / torch.sum(weights**2, dim=1)
            resample_mask = neff < (self.n_particles / 2)
            
            if resample_mask.any():
                b_idx = torch.where(resample_mask)[0]
                sampled_idx = torch.multinomial(weights[b_idx], self.n_particles, replacement=True)
                particles[b_idx] = torch.gather(
                    particles[b_idx], 1, sampled_idx.unsqueeze(-1).expand(-1, -1, self.x_dim)
                )
                weights[b_idx] = 1.0 / self.n_particles

        # 4. Extract trajectories using shared utility
        # CRITICAL: Move entire history to CPU ONCE to avoid sequential sync bottlenecks
        p_hist_cpu = p_hist.detach().cpu()
        w_hist_cpu = w_hist.detach().cpu()
        
        preds = track_modes(p_hist_cpu, w_hist_cpu, n_modes=self.n_modes)
        
        return preds.to(self.device)