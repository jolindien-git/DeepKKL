import os
import time
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.observer import BaseMultimodalObserver
from models.kkl_utils import KKL_Latent_Dynamics
from models.networks import TimeConditionedMLP
from models.kkl_cfm import KKL_CFM
from utils.tracking import track_modes_cfm


class KKLCFMObserver(BaseMultimodalObserver):
    """
    Uses Conditional Flow Matching to learn the multi-valued inverse mapping T^{-1}(z).
    Generates a distribution of possible physical states x(t).
    """
    def __init__(self, dataset, normalizer, z_dim=6, hidden_dim=128, n_layers=4, n_modes=2, n_steps=5, n_particles=50, device='cpu'):
        super().__init__(dataset.x_dim, dataset.y_dim, dataset.dt, device)
        
        self.normalizer = normalizer
        self.n_modes = n_modes 
        self.n_steps = n_steps
        
        self.latent_dyn = KKL_Latent_Dynamics(self.y_dim, z_dim, self.dt, device)
        v_network = TimeConditionedMLP(self.x_dim, z_dim, hidden_dim, n_layers).to(device)
        self.cfm = KKL_CFM(v_network).to(device)
        
        self.n_particles = n_particles

    def fit(self, train_xs, train_ys, epochs=100, batch_size=64*5, lr=1e-3, transient_len=50, transient_skip=True):
        """
        Trains the CFM vector field using Optimal Transport Flow Matching.
        """
        train_xs = train_xs.to(self.device)
        train_ys = train_ys.to(self.device)
        
        # 1. Compute z(t) over the full trajectory
        with torch.no_grad():
            zs_target = self.latent_dyn.compute_z_fast(train_ys)
            
        # 2. Skip transient if requested
        if transient_skip:
            xs_steady = train_xs[:, transient_len:]
            zs_steady = zs_target[:, transient_len:]
        else:
            xs_steady = train_xs
            zs_steady = zs_target
            
        # 3. Normalize the physical states for stable neural net training
        xs_norm = self.normalizer.normalize(xs_steady)
        
        # Flatten time and batch for standard OT Flow Matching
        xs_flat = xs_norm.reshape(-1, self.x_dim)
        zs_flat = zs_steady.reshape(-1, self.latent_dyn.z_dim)
        
        dataset = TensorDataset(xs_flat, zs_flat)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = torch.optim.Adam(self.cfm.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs//3), gamma=0.5)
        
        self.cfm.train()
        tic = time.time()
        for epoch in range(epochs):
            total_loss = 0
            for batch_x_norm, batch_z in loader:
                optimizer.zero_grad()
                
                # The magic happens here: CFM computes the OT path and MSE loss
                loss = self.cfm.compute_loss(batch_x_norm, batch_z)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            scheduler.step()
            period = max(1, epochs // 10)
            if epoch == 0 or (epoch + 1) % period == 0:
                elapsed = time.time() - tic
                print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(loader):.6f} - LR: {scheduler.get_last_lr()[0]:.6f} - Time: {elapsed:.2f}")
                
        print("--- Training Complete ---")

    @torch.no_grad()
    def forward(self, ys_batch):
        """
        Inference: Generates particles, and extracts distinct modes (tracking).
        Returns:
            tensor of shape (B, n_modes, T, x_dim).
        """
        # import time; tic = time.time() # test profiling
        self.cfm.eval()
        
        zs = self.latent_dyn.compute_z_fast(ys_batch)
        # torch.cuda.synchronize(); print('zs', time.time() - tic)
        
        # -- Generate N particles
        xs_norm_pred = self.cfm.sample(zs, n_particles=self.n_particles, n_steps=self.n_steps)
        # torch.cuda.synchronize(); print('cfm.sample', time.time() - tic)
        
        xs_pred = self.normalizer.unnormalize(xs_norm_pred) # Shape: (B, N, T, x_dim)
        
        # -- Rearrange dimensions to match the tracking utility signature
        xs_pred_aligned = xs_pred.permute(0, 2, 1, 3) # -> (B, T, N, D)
        
        # -- Apply tracking
        xs_tracked = track_modes_cfm(
            p_hist=xs_pred_aligned, 
            n_modes=self.n_modes
        )
        # print('track', time.time() - tic)
            
        return xs_tracked # (B, n_modes, T, x_dim)
    

    def save_model(self, path):
        """Saves the network weights and normalizer stats."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'v_network_state_dict': self.cfm.v_network.state_dict(),
            'normalizer_mean': self.normalizer.mean,
            'normalizer_std': self.normalizer.std
        }, path)
        print(f"[*] KKL-CFM saved to {path}")

    def load_model(self, path):
        """Loads the network weights and normalizer stats."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"No saved model found at {path}")
            
        checkpoint = torch.load(path, map_location=self.device)
        self.cfm.v_network.load_state_dict(checkpoint['v_network_state_dict'])
        
        # Restore normalizer state exactly as it was during training
        self.normalizer.mean = checkpoint['normalizer_mean'].to(self.device)
        self.normalizer.std = checkpoint['normalizer_std'].to(self.device)
        
        print(f"[*] KKL-CFM loaded from {path}")