import os
import sys
import argparse
import torch
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.systems import datasets
from utils.base import set_seed
from utils.metrics import compute_mhd_metrics
from utils.plotting import plot_model_comparison
from run_baselines import run_baseline


baselines_to_run = {
    'pf': 'BPF + Tracking',
    'nolcos': 'Set-Valued KKL',
    'node': 'Neural-ODE KKL',
    'cfm': 'Generative KKL'
}

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate all baselines.")
    parser.add_argument('--dataset', type=str, default='BiModal', choices=datasets.keys())
    parser.add_argument('--n_trajs_train', type=int, default=1000)
    parser.add_argument('--n_trajs_test', type=int, default=50)
    parser.add_argument('--traj_len', type=int, default=500)
    parser.add_argument('--noise_std', type=float, default=0)
    parser.add_argument('--seed', type=int, default=0)
    
    parser.add_argument('--z_dim', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--n_particles', type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("========== EVALUATING ALL BASELINES ==========")
    print(f"Dataset: {args.dataset} | Device: {device} | Test Trajs: {args.n_trajs_test}")
    print("==============================================")

    # 1. Setup Data
    set_seed(args.seed)
    dataset_cls = datasets[args.dataset]
    
    # Training data (for Neural-based baselines)
    train_dataset = dataset_cls(n_trajs=args.n_trajs_train, traj_len=args.traj_len, noise_std=args.noise_std)
    train_xs = torch.tensor(train_dataset.xs, dtype=torch.float32)
    train_ys = torch.tensor(train_dataset.ys, dtype=torch.float32)
    
    # Test data
    test_dataset = dataset_cls(n_trajs=args.n_trajs_test, traj_len=args.traj_len, noise_std=args.noise_std)
    ts = torch.tensor(test_dataset.ts[0], dtype=torch.float32)
    test_ys = torch.tensor(test_dataset.ys, dtype=torch.float32, device=device)
    test_xs = torch.tensor(test_dataset.xs, dtype=torch.float32, device=device)
    true_modes = test_dataset.get_true_modes(test_xs)
    
    # 3. Evaluation
    all_preds = {}
    all_observers = {}
    results = []
    for b_key, b_name in baselines_to_run.items():
        print(f"\n[{list(baselines_to_run.keys()).index(b_key)+1}/{len(baselines_to_run)}] Running {b_name}...")
        
        preds, inf_time, observer = run_baseline(
            baseline=b_key, 
            train_xs=train_xs, 
            train_ys=train_ys, 
            train_dataset=train_dataset, 
            test_ys=test_ys, 
            args=args, 
            device=device
        )
        
        # Store
        all_preds[b_name] = preds.cpu().numpy()
        all_observers[b_key] = observer
        p_val, c_val = compute_mhd_metrics(preds, true_modes, transient_len=args.traj_len // 10)
        results.append({'Model': b_key.upper(), 'Precision': p_val, 'Coverage': c_val, 'Inf. Time (s)': inf_time})

    # 4. Output and Visualization
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    
    df_results = pd.DataFrame(results).set_index('Model')
    print(df_results.to_string(float_format=lambda x: f"{x:.4f}"))
    
    csv_path = f"metrics_{args.dataset}_noise{args.noise_std}.csv"
    df_results.to_csv(csv_path)
    print(f"\nResults saved to {csv_path}")

    print("\nGenerating comparison plot...")
    plot_model_comparison(
        ts=ts.numpy(),
        ys=test_ys.cpu().numpy(),
        xs_true_modes=true_modes.cpu().numpy(),
        dict_preds=all_preds,
        batch_idx=0,
        save_path=f"comparison_{args.dataset}.png"
    )