"""
Particle Filter Observer for the 'Test' system.

Implements a Bootstrap Particle Filter (SIR) to estimate the state x from
noisy measurements y, for the same system used in the KKL-CFM project.

Noise parameters are kept consistent with main_KKL_CFM.py defaults:
    noise_std   = 0.0  (measurement noise std)
    process_std = 0.0  (process noise std)

You can override them via --noise_std and --process_std CLI args.

Visualization: temporal "probability tube" (percentile bands) per state dimension.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

from dataset import datasets, RK4


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# data
parser.add_argument('--dataset', type=str, default='Test')
parser.add_argument('--noise_std', type=float, default=1.)
parser.add_argument('--process_std', type=float, default=.0)
parser.add_argument('--traj_len', type=int, default=500)
# model
parser.add_argument('--n_particles', type=int,   default=2000,
                    help='Number of particles for the particle filter')
# train
parser.add_argument('--batch', type=int, default=0, help='Trajectory index to plot in detail')
parser.add_argument('--n_trajs', type=int, default=50, help='Number of test trajectories')
parser.add_argument('--resample_method', type=str, default='systematic', choices=['systematic', 'multinomial'],
                    help='Resampling strategy: systematic (lower variance) or multinomial')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
print(args)

np.random.seed(args.seed)

# ---------------------------------------------------------------------------
# Dataset / system
# ---------------------------------------------------------------------------
KKLDataset = datasets[args.dataset]
x_dim, y_dim = KKLDataset.x_dim, KKLDataset.y_dim
dt = KKLDataset.dt

dataset = KKLDataset(
    n_trajs=args.n_trajs,
    traj_len=args.traj_len,
    noise_std=args.noise_std,
    process_std=args.process_std,
)

# ts: (n_trajs, traj_len, 1)  ->  squeeze to (n_trajs, traj_len)
ts = dataset.ts[:, :, 0]        # (n_trajs, traj_len)
xs = dataset.xs                 # (n_trajs, traj_len, x_dim)
ys = dataset.ys                 # (n_trajs, traj_len, y_dim)


# ---------------------------------------------------------------------------
# Bootstrap Particle Filter
# ---------------------------------------------------------------------------

def particle_filter(ys_traj, n_particles, noise_std, process_std,
                    x0_low, x0_high, dt, get_derivs, get_y,
                    resample_method='systematic'):
    """
    Bootstrap SIR Particle Filter.

    Parameters
    ----------
    ys_traj     : (traj_len, y_dim)  — noisy measurements
    n_particles : int
    noise_std   : float — std of the measurement noise  N(0, noise_std**2)
    process_std : float — std of the process noise injected at each step
    x0_low, x0_high : (x_dim,)  — uniform prior on initial state
    dt          : float
    get_derivs  : callable(x, u=None) -> dx/dt   [numpy, shape (..., x_dim)]
    get_y       : callable(x)          -> y       [numpy, shape (..., y_dim)]
    resample_method  : 'systematic' or 'multinomial'

    Returns
    -------
    particles_history : (traj_len, n_particles, x_dim)
    weights_history   : (traj_len, n_particles)
    mean_estimate     : (traj_len, x_dim)
    resample_steps    : list of int

    """
    traj_len, y_dim = ys_traj.shape
    x_dim = len(x0_low)

    # --- measurement noise std: if 0 use a small default so weights are finite
    noise_std_clipped = noise_std if noise_std > 1e-8 else 1e-2

    # --- initialise particles uniformly in the prior
    particles = np.random.uniform(
        x0_low, x0_high, size=(n_particles, x_dim)
    ).astype(np.float32)
    weights = np.ones(n_particles, dtype=np.float32) / n_particles

    particles_history = np.zeros((traj_len, n_particles, x_dim), dtype=np.float32)
    weights_history = np.zeros((traj_len, n_particles), dtype=np.float32)
    resample_steps = []

    resample_fn = _systematic_resample if resample_method == 'systematic' \
        else _multinomial_resample

    for k in range(traj_len):
        y_k = ys_traj[k]  # (y_dim,)

        # ----- 1. Propagate (predict) -----
        if k > 0:
            noise = np.random.normal(0, process_std, size=particles.shape).astype(np.float32) \
                    if process_std > 1e-8 else 0.0
            particles = (
                RK4(get_derivs, dt, particles)
                + dt * noise
            ).astype(np.float32)

        # ----- 2. Update (weighting) -----
        y_pred = get_y(particles)              # (n_particles, y_dim)
        diff   = y_pred - y_k[np.newaxis, :]  # (n_particles, y_dim)
        # Isotropic Gaussian likelihood
        log_w  = -0.5 * np.sum(diff**2, axis=-1) / noise_std_clipped**2
        log_w -= log_w.max()                   # numerical stability
        weights = np.exp(log_w)
        weights /= weights.sum()

        # ----- 3. Store -----
        particles_history[k] = particles
        weights_history[k]   = weights

        # ----- 4. Resample (systematic resampling) -----
        neff = 1.0 / np.sum(weights**2)
        if neff < n_particles / 2:
            print("Resampling...")
            particles = resample_fn(particles, weights)
            weights = np.ones(n_particles, dtype=np.float32) / n_particles
            resample_steps.append(k)

    # Weighted mean estimate
    mean_estimate = np.einsum('tkd,tk->td', particles_history, weights_history)

    return particles_history, weights_history, mean_estimate, resample_steps


def _systematic_resample(particles, weights):
    """
    Systematic resampling — O(N), single random draw.
    Positions: (u + 0, u+1, ..., u+N-1) / N  with u ~ U[0,1).
    Lower variance than multinomial; standard choice in PF literature.
    """
    N = len(weights)
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0  # guard against float rounding
    positions = (np.arange(N) + np.random.uniform()) / N
    indices = np.searchsorted(cumsum, positions)
    indices = np.clip(indices, 0, N - 1)
    return particles[indices]


def _multinomial_resample(particles, weights):
    """
    Multinomial resampling — N independent draws from Categorical(weights).
    Simple but higher variance: a particle can be missed even if its weight is high.
    """
    N = len(weights)
    indices = np.random.choice(N, size=N, p=weights)
    return particles[indices]


# ---------------------------------------------------------------------------
# Run filter on every trajectory
# ---------------------------------------------------------------------------
print(f"Running Bootstrap Particle Filter  [N={args.n_particles}, "
      f"noise_std={args.noise_std}, process_std={args.process_std}]")

all_particles    = []   # list of (traj_len, n_particles, x_dim)
all_weights      = []   # list of (traj_len, n_particles)
all_means        = []   # list of (traj_len, x_dim)
all_resample_steps = [] # list of lists of int

for i in range(args.n_trajs):
    print(f"  trajectory {i+1}/{args.n_trajs} ...", end=" ", flush=True)
    ph, wh, me, rs = particle_filter(ys_traj=ys[i],
                                     n_particles=args.n_particles,
                                     noise_std=args.noise_std,
                                     process_std=args.process_std,
                                     x0_low=KKLDataset.x0_low,
                                     x0_high=KKLDataset.x0_high,
                                     dt=dt,
                                     get_derivs=KKLDataset.get_derivs,
                                     get_y=KKLDataset.get_y,
                                     resample_method=args.resample_method,)
    all_particles.append(ph)
    all_weights.append(wh)
    all_means.append(me)
    all_resample_steps.append(rs)

    mse = np.mean((me - xs[i])**2)
    print(f"MSE={mse:.4f}  RMSE={np.sqrt(mse):.4f}")

print("Done.")


# ---------------------------------------------------------------------------
# Plot  — probability tube for one trajectory
# ---------------------------------------------------------------------------

def compute_weighted_percentiles(particles, weights, percentiles):
    """
    particles : (traj_len, n_particles, x_dim)
    weights   : (traj_len, n_particles)
    percentiles: list of floats in [0, 100]

    Returns dict {pct: (traj_len, x_dim)}
    Each dimension is sorted independently before computing the weighted CDF.
    """
    traj_len, n_p, x_d = particles.shape
    result = {p: np.zeros((traj_len, x_d)) for p in percentiles}
    for t in range(traj_len):
        for d in range(x_d):
            sorted_idx = np.argsort(particles[t, :, d])
            w_sorted = weights[t, sorted_idx]
            p_sorted = particles[t, sorted_idx, d]
            cumw = np.cumsum(w_sorted)
            cumw /= cumw[-1]  # normalise to exactly 1
            for p in percentiles:
                idx = np.searchsorted(cumw, p / 100.0)
                idx = np.clip(idx, 0, n_p - 1)
                result[p][t, d] = p_sorted[idx]
    return result


def plot_probability_tube(b=0):
    sns.set(style="ticks", context="talk", font_scale=1.2)
    plt.rcParams["font.family"] = "serif"
    colors = sns.color_palette("Paired", n_colors=12)

    t  = ts[b]
    xb = xs[b]           # (traj_len, x_dim)
    yb = ys[b]           # (traj_len, y_dim)
    ph = all_particles[b]
    wh = all_weights[b]
    me = all_means[b]
    resample_times = t[all_resample_steps[b]] if all_resample_steps[b] else []

    pcts = compute_weighted_percentiles(ph, wh,
                                        percentiles=[5, 25, 50, 75, 95])

    fig, axes = plt.subplots(x_dim + 1, 1,
                             figsize=(12, 3 * (x_dim + 1)),
                             sharex=True)

    # --- top row: measurement y ---
    ax = axes[0]
    for j in range(y_dim):
        ax.plot(t, yb[:, j], color=colors[3+j], linewidth=1.2,
                label=f'$y_{j+1}$ (noisy)' if args.noise_std > 0 else f'$y_{j+1}$')
    for rt in resample_times:
        ax.axvline(rt, color=colors[9], linewidth=1.5, alpha=0.4)
    ax.set_ylabel('Output $y$')
    ax.legend(loc='right', fontsize=10)
    ax.grid(True, which='both', alpha=0.4)
    if args.noise_std > 0:
        ax.set_title(f"Particle Filter  —  {args.dataset}  "
                     f"(N={args.n_particles}, noise={args.noise_std}, "
                     f"{len(resample_times)} resamplings)")
    else:
        ax.set_title(f"Particle Filter  —  {args.dataset}  "
                     f"(N={args.n_particles}, noiseless, "
                     f"{len(resample_times)} resamplings)")

    # --- one row per state ---
    for i in range(x_dim):
        ax = axes[i + 1]

        # scatter: particle cloud
        t_expanded = np.repeat(t, args.n_particles)
        p_expanded = ph[:, :, i].flatten()
        ax.scatter(t_expanded, p_expanded, s=0.1, color='skyblue', alpha=0.05,
                   label='Particles')#, rasterized=True)

        # 90 % tube
        ax.fill_between(t,
                        pcts[5][:, i], pcts[95][:, i],
                        alpha=0.20, color=colors[5], label='90% interval')
        # # 50 % tube
        # ax.fill_between(t,
        #                 pcts[25][:, i], pcts[75][:, i],
        #                 alpha=0.40, color=colors[5], label='50% interval')
        # median
        ax.plot(t, pcts[50][:, i], color=colors[1], linewidth=1.5,
                linestyle='--', label='Median')
        # # weighted mean
        # ax.plot(t, me[:, i], color=colors[5], linewidth=1.5,
        #         label=r'$\hat x_{%i}$ (mean)' % (i + 1))
        # ground truth
        ax.plot(t, xb[:, i], color=colors[3], linewidth=1.2,
                linestyle='-', label='$x_{%i}$ (true)' % (i + 1))
        # resample instants
        for k, rt in enumerate(resample_times):
            ax.axvline(rt, color=colors[9], linewidth=1.5, alpha=0.4,
                       label='resample' if k == 0 and i == 0 else None)

        ax.set_ylabel(f'$x_{i+1}$')
        ax.legend(loc='right', fontsize=9)
        ax.grid(True, which='both', alpha=0.4)

    axes[-1].set_xlabel('time (s)')
    plt.tight_layout()
    plt.savefig('particle_filter_tube.pdf', bbox_inches='tight', format='pdf')
    plt.savefig('particle_filter_tube.png', dpi=150, bbox_inches='tight')

    print("Saved: particle_filter_tube.png")
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Plot  — MSE over time (averaged across trajectories)
# ---------------------------------------------------------------------------

def plot_mse_over_time():
    sns.set(style="ticks", context="talk", font_scale=1.2)
    plt.rcParams["font.family"] = "serif"
    colors = sns.color_palette("Paired", n_colors=10)

    mse_per_traj = []
    for i in range(args.n_trajs):
        err = (all_means[i] - xs[i])**2   # (traj_len, x_dim)
        mse_per_traj.append(err.mean(axis=-1))  # (traj_len,)
    mse_avg = np.mean(mse_per_traj, axis=0)     # (traj_len,)

    t = ts[0]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogy(t, mse_avg, color=colors[9],
                label='MSE (mean over %i trajs)' % args.n_trajs)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('MSE')
    ax.set_title(f'Particle Filter MSE  —  {args.dataset}')
    ax.legend()
    ax.grid(True, which='both', alpha=0.4)
    plt.tight_layout()
    plt.savefig('particle_filter_mse.png', dpi=150, bbox_inches='tight')
    print("Saved: particle_filter_mse.png")
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Run plots
# ---------------------------------------------------------------------------
plot_probability_tube(b=args.batch)
plot_mse_over_time()