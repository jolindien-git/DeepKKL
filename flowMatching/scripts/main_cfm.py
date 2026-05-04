import os
import sys
import argparse
import torch
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.systems import datasets
from models.kkl_cfm_observer import KKLCFMObserver
from utils.base import Normalizer, set_seed
from utils.metrics import compute_mhd_metrics
from utils.plotting import plot_multimodal_tracking, plot_density, plot_vector_field


def parse_args():
    parser = argparse.ArgumentParser()
    # -- data
    parser.add_argument('--dataset', type=str, default='BiModal', choices=datasets.keys(), help="Target dynamical system.")
    parser.add_argument('--noise_std', type=float, default=0, help="Measurement noise.")
    parser.add_argument('--traj_len', type=int, default=500, help="Length of each trajectory.")
    parser.add_argument('--n_trajs_train', type=int, default=1000, help="Number of training trajectories (for Nolcos/NODE).")
    parser.add_argument('--n_trajs_test', type=int, default=50, help="Number of testing trajectories.")
    parser.add_argument('--seed', type=int, default=0)
    # -- model KKL-CFM
    parser.add_argument('--z_dim', type=int, default=6)
    parser.add_argument('--n_particles', type=int, default=100, help="Partciles Number for eval.")
    parser.add_argument('--name', type=str, default='temp')
    # -- training
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128*4)
    parser.add_argument('--lr', type=float, default=2e-3)
    return parser.parse_args()


if __name__ == "__main__":
    
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    #%% Setup Data
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

    
    #%% Model
    set_seed(args.seed)
    normalizer = Normalizer(train_xs)
    observer = KKLCFMObserver(
        dataset=train_dataset, 
        normalizer=normalizer, 
        z_dim=args.z_dim, 
        hidden_dim=128,
        n_layers=4,
        n_modes=train_dataset.n_modes,
        n_particles=args.n_particles,
        device=device
    )
    
    
    #%% Train
    model_path = os.path.join("trained_models", args.dataset, f"{args.name}.pth")
    
    if args.epochs == 0 and os.path.exists(model_path):
        print("Loading pre-trained model...")
        observer.load_model(model_path)
    else:
        set_seed(args.seed)
        epochs = max(1, args.epochs)
        print(f"=== Training KKL-CFM on {args.dataset} ({device}) for {epochs} epochs===")
        observer.fit(
            train_xs, train_ys, 
            epochs=epochs, 
            batch_size=args.batch_size, 
            lr=args.lr,
            transient_len=args.traj_len // 10
        )
        print("Saving model : ", model_path)
        observer.save_model(model_path)
    
    
    #%% POST-TRAINING EVALUATION
    print("\n=== Running Post-Training Evaluation ===")
    test_dataset = dataset_cls(n_trajs=args.n_trajs_test, traj_len=args.traj_len, noise_std=args.noise_std)
    ts = torch.tensor(test_dataset.ts[0], dtype=torch.float32)
    test_ys = torch.tensor(test_dataset.ys, dtype=torch.float32, device=device)
    xs_test = torch.tensor(test_dataset.xs, dtype=torch.float32, device=device)
    true_modes = test_dataset.get_true_modes(xs_test)

    # Inference
    tic = time.time()
    preds_cfm = observer(test_ys) # Output shape: (B, M, T, x_dim)
    inf_time = time.time() - tic

    # Metrics
    p_cfm, c_cfm = compute_mhd_metrics(preds_cfm, true_modes, transient_len=args.traj_len//10)
    print(f"Precision : {p_cfm:.4f}")
    print(f"Coverage    : {c_cfm:.4f}")
    print(f"Inference Time       : {inf_time:.3f} s")

    #%% Plot 1: Multimodal Observer
    tracking_path = os.path.join("trained_models", args.dataset, f"{args.name}_track.png")
    plot_multimodal_tracking(
        ts=ts.numpy(), 
        ys=test_ys.cpu().numpy(), 
        xs_true_modes=true_modes.cpu().numpy(), 
        preds=preds_cfm.cpu().numpy(), 
        batch_idx=1,
        save_path=tracking_path
    )
    
    # %% Plot 2: Density Mapping at the end of the trajectory
    if test_dataset.x_dim == 2:
        print("Computing Density Map ...")
        
        batch_idx=0 # first traj
        taus=[0, .01, .1, 1]
        
        plot_density(
            observer=observer, 
            ys=test_dataset.ys[batch_idx],
            xs_true_modes=true_modes[batch_idx].cpu().numpy(), 
            t_idx=-1, # Evaluate at the very last time step
            grid_res=200,
            # taus=taus
        )
        
        # plot_vector_field(observer=observer, 
        #     ys=test_dataset.ys[batch_idx],
        #     xs_true_modes=true_modes[batch_idx].cpu().numpy(), 
        #     t_idx=-1, # Evaluate at the very last time step
        #     grid_res=200,
        #     taus=taus
        # )    