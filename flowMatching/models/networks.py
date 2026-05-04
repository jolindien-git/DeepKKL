import torch
import torch.nn as nn

from data.systems import RK4


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=[128, 128, 128], activation=nn.ReLU):
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(activation())
            current_dim = hidden_dim
            
        # Output layer (no activation)
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)
    


# === FLOW MATCHING ===

class TimeConditionedMLP(nn.Module):
    """
    Vector field v_theta(t, x | z) for Conditional Flow Matching.
    Predicts the derivative dx/dt along the Optimal Transport path.
    Remark : here t does not refer to the physical time of the system, but a virtual time \in [0, 1]
    """
    def __init__(self, x_dim, z_dim, hidden_dim=128, n_layers=4):
        super().__init__()
        
        # 1. Project the inputs
        # t_flow is 1D, x is x_dim
        self.time_x_proj = nn.Linear(1 + x_dim, hidden_dim)
        
        # The condition z is projected separately
        self.z_proj = nn.Linear(z_dim, hidden_dim)
        
        # 2. Hidden layers (Residual Blocks)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(), # SiLU (Swish) is standard for Score/Flow networks
                nn.Linear(hidden_dim, hidden_dim)
            ))
            
        # 3. Output layer (predicts dx/dt, so output dim is x_dim)
        self.out_proj = nn.Linear(hidden_dim, x_dim)
        
        # Initialize output layer to zero for stable early training
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, t_flow, x, z):
        """
        t_flow: (B, 1) or scalar
        x: (B, x_dim) or (B, T, x_dim) or (B, M, T, x_dim)
        z: (B, z_dim) or (B, T, z_dim) or (B, M, T, z_dim)
        """
        # 1. Convert purely scalar time to a 1D tensor
        if t_flow.dim() == 0:
            t_flow = t_flow.unsqueeze(0)
            
        # 2. Add missing dimensions to the right so t_flow matches x's dimensionality
        while t_flow.dim() < x.dim():
            t_flow = t_flow.unsqueeze(-1)
            
        # 3. Expand t_flow to match the exact batch/spatial shape of x
        t_flow = t_flow.expand_as(x[..., :1])
            
        # Now concatenation is 100% safe
        tx = torch.cat([t_flow, x], dim=-1)
        
        # 4. Process condition
        h_tx = self.time_x_proj(tx)
        h_z = self.z_proj(z)
        
        # Combine
        h = h_tx + h_z
        h = nn.functional.silu(h)
        
        # Pass through residual layers
        for layer in self.layers:
            h = h + layer(h) 
            
        v = self.out_proj(h)
        return v


# ==== NEURAL ODE =====
class ODEFunc(nn.Module):
    """ Vector field for Neural ODE. """
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, hidden_dim)
        )

    def forward(self, h, u=None):
        return self.net(h)


class NeuralODE_KKL(nn.Module):
    """ [Miao et al.] """
    def __init__(self, z_dim, x_dim, hidden_dim=128, n_steps=2):
        super().__init__()
        self.proj_in = nn.Linear(z_dim, hidden_dim)
        self.odefunc = ODEFunc(hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, x_dim)
        self.n_steps = n_steps

    def forward(self, z):
        # Project : z -> h
        h = self.proj_in(z)
        
        # Integrate (RK4) from tau=0 to tau=1
        dt_ode = 1.0 / self.n_steps
        for _ in range(self.n_steps):
            h = RK4(self.odefunc, dt_ode, h)
            
        # Project : h -> x
        x_hat = self.proj_out(h)
        return x_hat