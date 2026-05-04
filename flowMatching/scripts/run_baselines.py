import os
import sys
import argparse
import time
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.systems import datasets
from utils.metrics import compute_mhd_metrics
from utils.plotting import plot_multimodal_tracking
from utils.base import Normalizer, set_seed

# Import baselines
from models.baselines.particle_filter import ParticleFilterObserver
from models.baselines.invert_T_gradient import InvertTGradientObserver
from models.baselines.neural_ode import NeuralODEObserver
from models.kkl_cfm_observer import KKLCFMObserver


def parse_args():
    parser = argparse.ArgumentParser()
    # -- data
    parser.add_argument('--dataset', type=str, default='BiModal', choices=datasets.keys(), help="Target dynamical system.")
    parser.add_argument('--noise_std', type=float, default=0, help="Measurement noise.")
    parser.add_argument('--traj_len', type=int, default=500, help="Length of each trajectory.")
    parser.add_argument('--n_trajs_train', type=int, default=1000, help="Number of training trajectories (for Nolcos/NODE).")
    parser.add_argument('--n_trajs_test', type=int, default=50, help="Number of testing trajectories.")
    parser.add_argument('--seed', type=int, default=0)
    # -- baseline parameters
    parser.add_argument('--baseline', type=str, required=True, choices=['pf', 'nolcos', 'node', 'cfm'], help="Baseline to evaluate.")
    parser.add_argument('--z_dim', type=int, default=6, help="for KKL-based models")
    parser.add_argument('--epochs', type=int, default=20, help="for neural-based models")
    parser.add_argument('--n_particles', type=int, default=100, help="Particles Number (for Particle-based models).")
    # -- plot result ?
    parser.add_argument('--plot', action='store_true', help="Plot the first trajectory tracking.")
    return parser.parse_args()


def run_baseline(baseline, train_xs, train_ys, train_dataset, test_ys, args, device):
    """
    Instantiates, fits, and runs inference for a selected model.
    Returns: preds, inference_time, observer_instance
    """
    set_seed(args.seed)
    transient_len = args.traj_len // 10

    # --- 1. Instantiation & Fit ---
    if baseline == 'pf':
        observer = ParticleFilterObserver(
            dataset=train_dataset, 
            n_particles=args.n_particles, 
            n_modes=train_dataset.n_modes, 
            noise_std=max(.1, train_dataset.noise_std),
            device=device
        )
        
    elif baseline == 'nolcos':
        observer = InvertTGradientObserver(
            dataset=train_dataset, 
            z_dim=args.z_dim, 
            n_opt=20, 
            lr=1.0, 
            device=device
        )
        set_seed(args.seed)
        observer.fit(train_xs, train_ys, epochs=args.epochs, transient_len=transient_len)
        
    elif baseline == 'node':
        normalizer = Normalizer(train_xs)
        observer = NeuralODEObserver(
            dataset=train_dataset, 
            z_dim=args.z_dim, 
            hidden_dim=128, 
            device=device
        )
        set_seed(args.seed)
        observer.fit(train_xs, train_ys, epochs=args.epochs, transient_len=transient_len)
        
    elif baseline == 'cfm':
        normalizer = Normalizer(train_xs)
        observer = KKLCFMObserver(
            dataset=train_dataset, 
            normalizer=normalizer, 
            z_dim=args.z_dim, 
            hidden_dim=128, 
            n_layers=4,
            n_particles=args.n_particles,
            n_modes=train_dataset.n_modes,
            device=device
        )
        set_seed(args.seed)
        observer.fit(train_xs, train_ys, epochs=args.epochs, batch_size=128*4, lr=2e-3, transient_len=transient_len)
        
    else:
        raise ValueError(f"Unknown baseline: {baseline}")

    # --- 2. Inference ---
    tic = time.time()
    with torch.no_grad() if baseline != 'nolcos' else torch.enable_grad():
        preds = observer(test_ys)
    inf_time = time.time() - tic

    return preds, inf_time, observer


if __name__ == "__main__":
    
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- Running {args.baseline} on {args.dataset} dataset ({device}) ---")
    
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
    xs_batch = torch.tensor(test_dataset.xs, dtype=torch.float32, device=device)
    true_modes = test_dataset.get_true_modes(xs_batch)

    # 2. Init / Train / Infer
    preds, inf_time, observer = run_baseline(
        args.baseline, train_xs, train_ys, train_dataset, test_ys, args, device
    )

    # 3. Evaluate (Metrics)
    precision, coverage = compute_mhd_metrics(preds, true_modes, transient_len=args.traj_len // 10)
    print("\n--- Final Results ---")
    print(f"Precision (Validity) : {precision:.4f}")
    print(f"Coverage (mode collapse)    : {coverage:.4f}")
    print(f"Inf. Time (s)    : {inf_time:.3f}")

    # 4. Plot ?
    if args.plot:
        plot_multimodal_tracking(
            ts=ts.numpy(), 
            ys=test_ys.cpu().numpy(), 
            xs_true_modes=true_modes.cpu().numpy(), 
            preds=preds.cpu().numpy(), 
            batch_idx=0,
        )
