import torch.nn as nn

class BaseMultimodalObserver(nn.Module):
    """
    Base class for all multimodal observers.
    Enforces a standard output format: (B, M_pred, T, D) for unified metric computation.
    """
    def __init__(self, x_dim, y_dim, dt, device='cpu'):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.dt = dt
        self.device = device

    def forward(self, ys_batch, n_particles=None):
        """
        Must be implemented by subclasses.
        
        Args:
            ys_batch: torch.Tensor of shape (B, T, y_dim)
            n_particles (optional): int or None - for particle based estimators
            
        Returns:
            preds: torch.Tensor of shape (B, M_pred, T, x_dim)
        """
        raise NotImplementedError("The forward method must be implemented.")