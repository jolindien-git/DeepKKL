import torch

def compute_mhd_metrics(preds, truths, transient_len=0):
    """
    Computes Precision (Validity) and Coverage (Recall) using Modified Hausdorff Distance.
    
    Args:
        preds: Tensor (B, M_pred, T, D) - Estimated modes
        truths: Tensor (B, C_true, T, D) - Ground truth branches
        transient: int - Slice out the transient part
        
    Returns:
        precision: float - Average error from preds to truths
        coverage: float - Average error from truths to preds
    """
    # Slice out the transient part
    if transient_len:
        preds = preds[:, :, transient_len:]
        truths = truths[:, :, transient_len:]
    
    # Swap T and M/C dimensions to compute distances per time step: (B, T, M, D)
    preds = preds.transpose(1, 2)
    truths = truths.transpose(1, 2)
    
    # Pairwise L2 distances: (B, T, M_pred, C_true)
    dists = torch.cdist(preds, truths, p=2.0)
    
    # Precision: For each pred mode, find closest true mode, average over pred modes
    precision = dists.min(dim=3)[0].mean(dim=2).mean().item()
    
    # Coverage: For each true mode, find closest pred mode, average over true modes
    coverage = dists.min(dim=2)[0].mean(dim=2).mean().item()
    
    return precision, coverage