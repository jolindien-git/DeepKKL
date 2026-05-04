import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

def plot_multimodal_tracking(ts, ys, xs_true_modes, preds, batch_idx=0, save_path=None):
    """
    Visualizes the true indistinguishable branches vs. the tracked estimates.
    
    Args:
        ts: Array/Tensor (T,)
        ys: Array/Tensor (B, T, Y_dim)
        xs_true_modes: Array/Tensor (B, C, T, X_dim)
        preds: Array/Tensor (B, M, T, X_dim)
        batch_idx: Index of the trajectory to plot
    """
    sns.set_theme(style="ticks", context="talk", font_scale=1.2)
    plt.rcParams["font.family"] = "serif"
    colors = sns.color_palette("Paired", n_colors=10)
    
    t = ts
    y_b = ys[batch_idx]
    x_true_b = xs_true_modes[batch_idx] # (C, T, X_dim)
    preds_b = preds[batch_idx]          # (M, T, X_dim)
    
    C_true, _, x_dim = x_true_b.shape
    M_pred = preds_b.shape[0]
    
    fig, axes = plt.subplots(x_dim + 1, 1, figsize=(10, 2.5 * (x_dim + 1)), sharex=True)
    
    # --- Plot Observations ---
    y_dim = y_b.shape[-1]
    for j in range(y_dim):
        axes[0].plot(t, y_b[:, j], color=colors[3+j], label=f'$y_{j+1}$')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # --- Plot States ---
    for i in range(x_dim):
        ax = axes[i+1]
        
        # Ground Truths
        # for c in range(1):
        #     ax.plot(t, x_true_b[c, :, i], linewidth=3, color=colors[3], 
        #             label='$x_%i$'%(i+1,))
        for c in range(C_true):
            ax.plot(t, x_true_b[c, :, i], linewidth=3, color=colors[3], 
                    label='$x_%i$'%(i+1,) if c==0 else "")
            
        # Predictions (Dashed lines)
        for m in range(M_pred):
            ax.plot(t, preds_b[m, :, i], '--', linewidth=2, color=colors[5+m], 
                    label='$\hat x_%i^{(%i)}$' % (i+1, m+1))
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
    axes[-1].set_xlabel('time (s)')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_model_comparison(ts, ys, xs_true_modes, dict_preds, batch_idx=0, save_path=None):
    """
    Plots a side-by-side comparison of multiple models for a single trajectory.
    
    Args:
        ts: Array (T,)
        ys: Array (B, T, Y_dim)
        xs_true_modes: Array (B, C, T, X_dim)
        dict_preds: Dictionary { 'Model Name': Array (B, M, T, X_dim) }
        batch_idx: Index of the trajectory to plot
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    sns.set_theme(style="ticks", context="paper", font_scale=1.2)
    colors = sns.color_palette("Paired", n_colors=12)
    
    t = ts
    x_true_b = xs_true_modes[batch_idx] 
    C_true, _, x_dim = x_true_b.shape
    n_models = len(dict_preds)
    
    fig, axes = plt.subplots(x_dim, n_models, figsize=(4 * n_models, 2.5 * x_dim), sharex=True, sharey='row')
    
    if x_dim == 1:
        axes = np.expand_dims(axes, axis=0)
    
    for col_idx, (model_name, preds) in enumerate(dict_preds.items()):
        preds_b = preds[batch_idx]
        M_pred = preds_b.shape[0]
        
        for i in range(x_dim):
            ax = axes[i, col_idx]
            
            # Plot Ground Truths
            for c in range(C_true):
                ax.plot(t, x_true_b[c, :, i], linewidth=3, color=colors[3], alpha=0.5,
                        label='True Modes' if c == 0 and i == 0 and col_idx == 0 else "")
                
            # Plot Predictions
            for m in range(M_pred):
                ax.plot(t, preds_b[m, :, i], '--', linewidth=2, color=colors[5+m], 
                        label=f'Pred Mode {m+1}' if i == 0 and col_idx == 0 else "")
                
            if i == 0:
                ax.set_title(model_name, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(f'State $x_{i+1}$')
            if i == x_dim - 1:
                ax.set_xlabel('Time (s)')
                
            ax.grid(True, alpha=0.3)
            
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=C_true+M_pred, frameon=False)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    plt.show()


def plot_density(observer, ys, xs_true_modes, taus=[1.0], t_idx=-1, grid_res=100, save_path=None):
    """
    Plots the density map p(x|z) at a specific time step.
    Args:
        ys: Tensor (T, y_dim)
        xs_true_modes: Tensor (n_modes, T, x_dim)
        taus: list - virtual times \in [0, 1]
    """
    
    device = observer.device
    
    # -- Densities
    densities = []
    observer.cfm.eval()
    with torch.no_grad():
        # -- compute z
        ys = torch.tensor(ys, dtype=torch.float32, device=device)
        z_traj = observer.latent_dyn.compute_z_fast(ys.unsqueeze(0))
        z_cond = z_traj[:, t_idx, :] # (1, z_dim)
        
        # -- compute mapping bounds
        x_max_val = np.abs(xs_true_modes).max(axis=(0, 1))
        margin = 1.5
        x_min_bounds = -x_max_val * margin
        x_max_bounds = x_max_val * margin
        
        # -- call Density computation
        for tau in taus:
            X1, X2, Density = observer.cfm.compute_density_map(
                z_cond, x_min_bounds, x_max_bounds, observer.normalizer,
                grid_res, 20, tau
            )
            densities.append(Density)
    
    # -- plot
    n_plots = len(densities)
    if n_plots == 1:
        plt.figure(figsize=(8, 6))
        Density_norm = Density/(Density.max())
        plt.contourf(X1, X2, Density_norm, levels=50, cmap='viridis')
        cbar = plt.colorbar()
        cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        
        # -- plot x_true
        # x_true_t = xs_true_modes[:, t_idx, :] 
        # for c in range(x_true_t.shape[0]):
        #     plt.scatter(x_true_t[c, 0], x_true_t[c, 1], color='red', s=100, marker='X', 
        #                 edgecolors='white', label='True Modes' if c==0 else "")
        # plt.legend(); 
        # plt.tight_layout()
    else:
        plt.figure(figsize=(8 * n_plots, 8))
        for i in range(n_plots):
            plt.subplot(1, n_plots, i + 1)
            plt.xticks([]); plt.yticks([])
            plt.contourf(X1, X2, densities[i], levels=50, cmap='viridis')
        plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Density plot saved to {save_path}")
    plt.show()


def plot_vector_field(observer, ys, xs_true_modes, taus=[1.0], t_idx=-1, grid_res=25, save_path=None):
    """
    Visualizes the learned vector field v(x, tau, z) across multiple flow times.
    """
    
    device = observer.device
    
    # -- Compute vector fields
    fields = []
    observer.cfm.eval()
    with torch.no_grad():
        # -- compute z condition[cite: 6, 10]
        ys_t = torch.tensor(ys, dtype=torch.float32, device=device)
        z_traj = observer.latent_dyn.compute_z_fast(ys_t.unsqueeze(0))
        z_cond = z_traj[:, t_idx, :] # (1, z_dim)[cite: 10]
        
        # -- compute mapping bounds[cite: 10]
        x_max_val = np.abs(xs_true_modes).max(axis=(0, 1))
        margin = 1.5
        x_min_bounds = -x_max_val * margin
        x_max_bounds = x_max_val * margin
        
        # -- grid preparation
        x_lin1 = np.linspace(x_min_bounds[0], x_max_bounds[0], grid_res)
        x_lin2 = np.linspace(x_min_bounds[1], x_max_bounds[1], grid_res)
        X1, X2 = np.meshgrid(x_lin1, x_lin2)
        x_flat = np.vstack([X1.ravel(), X2.ravel()]).T
        x_torch = torch.FloatTensor(x_flat).to(device)
        
        # -- normalize grid points[cite: 9]
        x_norm = observer.normalizer.normalize(x_torch)
        z_batch = z_cond.expand(x_norm.shape[0], -1)
        
        for tau in taus:
            tau_batch = torch.ones(x_norm.shape[0], 1, device=device) * tau
            # Get normalized velocity[cite: 4]
            v_norm = observer.cfm.v_network(tau_batch, x_norm, z_batch)
            # Rescale to physical space velocity[cite: 9]
            v_phys = v_norm * observer.normalizer.std.to(device)
            
            U = v_phys[:, 0].reshape(grid_res, grid_res).cpu().numpy()
            V = v_phys[:, 1].reshape(grid_res, grid_res).cpu().numpy()
            fields.append((U, V))

    # -- plot
    n_plots = len(fields)
    if n_plots == 1:
        plt.figure(figsize=(8, 6))
        U, V = fields[0]
        plt.streamplot(X1, X2, U, V, color=(0, 0, 1, 0.3), density=1.5)
        plt.title(f"Vector Field (tau={taus[0]})")
    else:
        plt.figure(figsize=(8 * n_plots, 8))
        for i in range(n_plots):
            plt.subplot(1, n_plots, i + 1)
            plt.xticks([]); plt.yticks([])
            U, V = fields[i]
            plt.streamplot(X1, X2, U, V, color=(0, 0, 1, 0.3), density=1.5)
            plt.title(f"tau={taus[i]}")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Vector field plot saved to {save_path}")
    plt.show()