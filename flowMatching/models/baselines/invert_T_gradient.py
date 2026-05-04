import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models.observer import BaseMultimodalObserver
from models.networks import SimpleMLP
from models.kkl_utils import KKL_Latent_Dynamics


class InvertTGradientObserver(BaseMultimodalObserver):
    """
    Set-Valued KKL Baseline (P. Bernard).
    Strictly unimodal observer. Tracks a single topological branch 
    based on the initial random guess at t=0.
    """
    def __init__(self, dataset, z_dim=6, n_opt=20, lr=1.0, device='cpu'):
        super().__init__(dataset.x_dim, dataset.y_dim, dataset.dt, device)
        
        # Hardcoded to 1 because this baseline is fundamentally unimodal
        self.n_modes = 1 
        self.n_opt = n_opt
        self.lr = lr
        
        self.x0_low = torch.tensor(dataset.x0_low, device=device)
        self.x0_high = torch.tensor(dataset.x0_high, device=device)
        
        self.latent_dyn = KKL_Latent_Dynamics(self.y_dim, z_dim, self.dt, device)
        self.trained_T = SimpleMLP(self.x_dim, z_dim, hidden_layers=[128, 128, 128]).to(device)

    def fit(self, train_xs, train_ys, epochs=50, batch_size=320, lr=1e-3, transient_len=50):
        """
        Trains the T(x) = z map on the fly.
        """
        print(f"--- Training T(x)=z Network for {epochs} epochs ---")
        
        train_xs = train_xs.to(self.device)
        train_ys = train_ys.to(self.device)
        
        # Compute z(t) over the full trajectory to respect observer memory
        with torch.no_grad():
            zs_target = self.latent_dyn.compute_z_fast(train_ys)
            
        # Slice out the transient part for mapping T(x) ≈ z(x)
        xs_steady = train_xs[:, transient_len:]
        zs_steady = zs_target[:, transient_len:]
        
        xs_flat = xs_steady.reshape(-1, self.x_dim)
        zs_flat = zs_steady.reshape(-1, self.latent_dyn.z_dim)
        
        dataset = TensorDataset(xs_flat, zs_flat)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = torch.optim.Adam(self.trained_T.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        self.trained_T.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_z in loader:
                optimizer.zero_grad()
                z_pred = self.trained_T(batch_x)
                loss = criterion(z_pred, batch_z)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            period = max(1, epochs // 10)
            if (epoch + 1) % period == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(loader):.6f}")
                
        print("--- Training Complete ---")

    def forward(self, ys_batch):
        """
        Inference: Invert T(x)=z locally over time for a single trajectory.
        Returns tensor of shape (B, 1, T, x_dim).
        """
        self.trained_T.eval()
        B, T_len, _ = ys_batch.shape
        
        with torch.no_grad():
            zs_target = self.latent_dyn.compute_z_fast(ys_batch)
            
        # Initialize 1 random candidate per batch element: (B, 1, D)
        x_curr_data = torch.empty((B, 1, self.x_dim), device=self.device)
        for d in range(self.x_dim):
            x_curr_data[..., d].uniform_(self.x0_low[d].item(), self.x0_high[d].item())
            
        xs_obs = torch.zeros((B, 1, T_len, self.x_dim), device=self.device)

        # Step-by-step optimization using Warm Start
        for t in range(T_len):
            z_t = zs_target[:, t, :].unsqueeze(1) 
            
            x_param = nn.Parameter(x_curr_data)
            optimizer = torch.optim.LBFGS(
                [x_param], 
                lr=self.lr, 
                max_iter=self.n_opt, 
                line_search_fn="strong_wolfe"
            )
            
            def closure():
                optimizer.zero_grad()
                z_pred = self.trained_T(x_param) 
                loss = torch.mean((z_pred - z_t)**2)
                loss.backward()
                return loss
                
            optimizer.step(closure)
            
            x_curr_data = x_param.detach().clone()
            xs_obs[:, :, t, :] = x_curr_data
            
        return xs_obs