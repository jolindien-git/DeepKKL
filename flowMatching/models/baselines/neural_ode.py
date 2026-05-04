import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models.observer import BaseMultimodalObserver
from models.networks import NeuralODE_KKL
from models.kkl_utils import KKL_Latent_Dynamics

class NeuralODEObserver(BaseMultimodalObserver):
    """
    Neural ODE KKL Baseline.
    Learns the deterministic inverse mapping z -> x globally using a Neural ODE.
    Strictly unimodal observer (n_modes = 1).
    """
    def __init__(self, dataset, z_dim=6, hidden_dim=128, n_steps=2, device='cpu'):
        super().__init__(dataset.x_dim, dataset.y_dim, dataset.dt, device)
        
        self.n_modes = 1 # Deterministic unimodal output
        
        self.latent_dyn = KKL_Latent_Dynamics(self.y_dim, z_dim, self.dt, device)
        self.node_model = NeuralODE_KKL(z_dim, self.x_dim, hidden_dim, n_steps).to(device)

    def fit(self, train_xs, train_ys, epochs=50, batch_size=320, lr=2e-3, transient_len=0):
        """
        Trains the Neural ODE inversion mapping z -> x.
        """
        print(f"--- Training Neural ODE KKL for {epochs} epochs ---")
        train_xs = train_xs.to(self.device)
        train_ys = train_ys.to(self.device)
        
        # 1. Compute z(t) over the full trajectory
        with torch.no_grad():
            zs_target = self.latent_dyn.compute_z_fast(train_ys)
            
        # 2. Skip transient if requested
        xs_steady = train_xs[:, transient_len:]
        zs_steady = zs_target[:, transient_len:]
            
        
        xs_flat = xs_steady.reshape(-1, self.x_dim)
        zs_flat = zs_steady.reshape(-1, self.latent_dyn.z_dim)
        
        dataset = TensorDataset(zs_flat, xs_flat)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = torch.optim.Adam(self.node_model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs//4), gamma=0.5)
        criterion = nn.MSELoss()
        
        self.node_model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_z, batch_x in loader:
                optimizer.zero_grad()
                x_pred = self.node_model(batch_z)
                loss = criterion(x_pred, batch_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            scheduler.step()
            period = max(1, epochs // 10)
            if (epoch + 1) % period == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(loader):.6f} - LR: {scheduler.get_last_lr()[0]:.6f}")
                
        print("--- Training Complete ---")

    def forward(self, ys_batch):
        """
        Inference: Maps observations y(t) to state x(t) via Neural ODE.
        Returns tensor of shape (B, 1, T, x_dim).
        """
        self.node_model.eval()
        with torch.no_grad():
            # 1. Generate latent trajectory
            zs = self.latent_dyn.compute_z_fast(ys_batch)
            
            # 2. Forward through Neural ODE
            xs_obs = self.node_model(zs)
            
        # Add the 'M' dimension to respect (B, M, T, D) format
        return xs_obs.unsqueeze(1)