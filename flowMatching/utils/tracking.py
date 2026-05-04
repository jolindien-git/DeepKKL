import torch
from scipy.optimize import linear_sum_assignment 

def track_modes(p_hist, w_hist=None, n_modes=2, n_kmeans_iter=5):
    """
    Batched K-Means + Hungarian tracking.
    
    Args:
        p_hist: Tensor (B, T, N, D)
        w_hist: Tensor (B, T, N) or None
        n_modes: int
        
    Returns:
        preds: Tensor (B, M, T, D)
    """
    B, T_len, N, D = p_hist.shape
    device = p_hist.device
    preds = torch.zeros((B, n_modes, T_len, D), device=device)
    
    if w_hist is None:
        w_hist = torch.ones((B, T_len, N), device=device) / N

    prev_centroids = None
    
    for t in range(T_len):
        pts = p_hist[:, t]  
        wts = w_hist[:, t]  
        
        if prev_centroids is not None:
            centroids = prev_centroids.clone()
        else:
            centroids = torch.zeros((B, n_modes, D), device=device)
            for m in range(n_modes):
                idx = torch.randint(0, N, (B,), device=device)
                centroids[:, m] = pts[torch.arange(B), idx]
                
        # Vectorized K-Means (No batch loops)
        for _ in range(n_kmeans_iter):
            dists = torch.cdist(pts, centroids)
            labels = torch.argmin(dists, dim=2) 
            
            new_centroids = torch.zeros_like(centroids)
            for k in range(n_modes):
                mask = (labels == k).float()
                mask_w = mask * wts
                w_sum = mask_w.sum(dim=1, keepdim=True)
                
                sum_pts = (pts * mask_w.unsqueeze(-1)).sum(dim=1)
                valid = (w_sum.squeeze(-1) > 1e-10)
                
                new_centroids[valid, k] = sum_pts[valid] / w_sum[valid]
                new_centroids[~valid, k] = centroids[~valid, k]
                
            centroids = new_centroids
            
        # Hungarian Matching
        if prev_centroids is not None:
            cost_matrices = torch.cdist(prev_centroids, centroids).cpu().numpy() 
            for b in range(B):
                _, col_ind = linear_sum_assignment(cost_matrices[b])
                centroids[b] = centroids[b, col_ind] 
                
        preds[:, :, t, :] = centroids
        prev_centroids = centroids
        
    return preds


def track_modes_cfm(p_hist, n_modes=2, n_iter=5):
    """
    Tracking ultra-rapide dédié au KKL-CFM (sans pondération).
    Gère les clusters vides en gardant la position précédente.
    100% Vectorisé sur le GPU.
    """
    B, T_len, N, D = p_hist.shape
    device = p_hist.device
    preds = torch.zeros((B, n_modes, T_len, D), device=device)

    prev_centroids = None

    for t in range(T_len):
        pts = p_hist[:, t]  # (B, N, D)

        if prev_centroids is not None:
            centroids = prev_centroids.clone()
        else:
            # Initialisation aléatoire
            indices = torch.randint(0, N, (B, n_modes), device=device)
            batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, n_modes)
            centroids = pts[batch_idx, indices]

        # --- Vectorized Unweighted K-Means ---
        for _ in range(n_iter):
            dists = torch.cdist(pts, centroids) # (B, N, M)
            labels = torch.argmin(dists, dim=2) # (B, N)

            # Masque d'assignation (B, N, M)
            mask = torch.nn.functional.one_hot(labels, num_classes=n_modes).float()
            
            # Comptage des particules par cluster (B, 1, M)
            counts = mask.sum(dim=1, keepdim=True)
            
            # Somme des positions (B, M, D)
            sum_pts = torch.bmm(mask.transpose(1, 2), pts)
            
            # Masque de validité : le cluster contient-il au moins 1 particule ? (B, M, 1)
            valid = (counts > 0).transpose(1, 2)

            # Mise à jour : Moyenne si valide, sinon on garde l'ancienne position (centroids)
            centroids = torch.where(valid, sum_pts / counts.transpose(1, 2).clamp(min=1), centroids)

        # --- Fast GPU Matching (Temporal Consistency) ---
        if prev_centroids is not None:
            cost_matrices = torch.cdist(prev_centroids, centroids) # (B, M, M)
            
            if n_modes == 2:
                # 100% Vectorisé pour 2 modes (Évite totalement scipy et le CPU)
                cost_straight = cost_matrices[:, 0, 0] + cost_matrices[:, 1, 1]
                cost_crossed = cost_matrices[:, 0, 1] + cost_matrices[:, 1, 0]
                
                swap = cost_crossed < cost_straight # (B,)
                
                temp = centroids.clone()
                centroids[swap, 0] = temp[swap, 1]
                centroids[swap, 1] = temp[swap, 0]
            else:
                # Fallback pour N > 2 modes
                from scipy.optimize import linear_sum_assignment
                cost_cpu = cost_matrices.cpu().numpy()
                for b in range(B):
                    _, col_ind = linear_sum_assignment(cost_cpu[b])
                    centroids[b] = centroids[b, col_ind]

        preds[:, :, t, :] = centroids
        prev_centroids = centroids

    return preds