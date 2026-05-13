import os
import sys
import time
import torch
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.systems import datasets
from utils.base import set_seed
from run_baselines import run_baseline
from utils.metrics import compute_mhd_metrics
from evaluate_all import parse_args
from main_cfm import Normalizer, KKLCFMObserver, get_model_path

N_PARTICLES_PF = [16, 64, 256, 1024, 2048]
N_PARTICLES_CFM = [8, 16, 64, 256, 1024]
MODEL_NAME = "noisy" # KKL-CFM model name (cf. main_cfm.py to train)


# %% read args
args = parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# %% Setup Data

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

# %% Evaluation Particle filter
print("========= Run Particle Filter", args.dataset)
pf_prec, pf_cov = [], []
for n_particles in N_PARTICLES_PF:
    args.n_particles = n_particles
    preds, inf_time, observer = run_baseline(
        baseline="pf", 
        train_xs=train_xs, 
        train_ys=train_ys, 
        train_dataset=train_dataset, 
        test_ys=test_ys, 
        args=args, 
        device=device
    )
    p_val, c_val = compute_mhd_metrics(preds, true_modes, transient_len=args.traj_len // 10)
    pf_prec.append(p_val)
    pf_cov.append(c_val)
    print("Particle Filter", 
          'n_particles', n_particles,
          "Prec.", p_val,
          "Cov.", p_val,
           "Time", inf_time,
          )

# %% Evaluation Particle filter
print("========= Run Generative KKL", args.dataset)
normalizer = Normalizer(train_xs)
observer = KKLCFMObserver(
    dataset=train_dataset, 
    normalizer=normalizer, 
    z_dim=12 if args.dataset == 'VDP2' else 6,# args.z_dim, 
    hidden_dim=128,
    n_layers=4,
    n_modes=train_dataset.n_modes,
    n_particles=args.n_particles,
    device=device
)
model_path = get_model_path(args.dataset, MODEL_NAME)
observer.load_model(model_path)

cfm_prec, cfm_cov = [], []
for n_particles in N_PARTICLES_CFM:
    observer.n_particles = n_particles
    
    tic = time.time()
    preds = observer(test_ys) # Output shape: (B, M, T, x_dim)
    inf_time = time.time() - tic
    p_val, c_val = compute_mhd_metrics(preds, true_modes, transient_len=args.traj_len // 10)
    cfm_prec.append(p_val)
    cfm_cov.append(c_val)
    print(args.dataset, 
          'n_particles',n_particles,
          "Prec.", p_val,
          "Cov.", p_val,
           "Time", inf_time,
          )

# %% plot
plt.semilogx(N_PARTICLES_PF, pf_prec, 'r*-', label='BPF Prec.')
plt.semilogx(N_PARTICLES_PF, pf_cov, 'r*--', label='BPF Cov.')
plt.semilogx(N_PARTICLES_CFM, cfm_prec, 'b*-', label='KKL-CFM Prec.')
plt.semilogx(N_PARTICLES_CFM, cfm_cov, 'b*--', label='KKL-CFM Cov.')
plt.legend()
plt.xlabel("N particles")
plt.show()