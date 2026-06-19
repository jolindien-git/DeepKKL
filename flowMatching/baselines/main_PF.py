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
from scipy.optimize import linear_sum_assignment

from dataset import datasets, RK4

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Test')
parser.add_argument('--noise_std', type=float, default=0.1,
                    help='Measurement noise std (matches KKL-CFM default)')
parser.add_argument('--process_std', type=float, default=0.0,
                    help='Process noise std (matches KKL-CFM default)')
parser.add_argument('--traj_len', type=int, default=500,
                    help='Trajectory length (time-steps)')
parser.add_argument('--n_particles', type=int, default=2000,
                    help='Number of particles')
parser.add_argument('--batch', type=int, default=0,
                    help='Trajectory index to plot in detail')
parser.add_argument('--n_trajs', type=int, default=50,
                    help='Number of test trajectories')
parser.add_argument('--resample_method', type=str, default='systematic',
                    choices=['systematic', 'multinomial'],
                    help='Resampling strategy: systematic (lower variance) or multinomial')
parser.add_argument('--n_modes', type=int, default=2,
                    help='Number of modes for k-means clustering of particles. '
                         '1 = disabled (default behaviour: weighted median).')
parser.add_argument('--kmeans_init', type=str, default='kmeanspp',
                    choices=['random', 'kmeanspp'],
                    help="K-means initialization: 'random' or 'kmeanspp'")
parser.add_argument('--pf_artificial_w_noise_std', type=float, default=0.01,
                    help="Artificial process noise added to the PF at each step")
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
print(args)
np.random.seed(args.seed)

# ---------------------------------------------------------------------------
# Dataset / system
# ---------------------------------------------------------------------------
KKLDataset = datasets[args.dataset]
x_dim = KKLDataset.x_dim
y_dim = KKLDataset.y_dim
dt = KKLDataset.dt

dataset = KKLDataset(
    n_trajs=args.n_trajs,
    traj_len=args.traj_len,
    noise_std=args.noise_std,
    process_std=args.process_std,
)

# ts: (n_trajs, traj_len, 1)  ->  squeeze to (n_trajs, traj_len)
ts = dataset.ts[:, :, 0]  # (n_trajs, traj_len)
xs = dataset.xs  # (n_trajs, traj_len, x_dim)
ys = dataset.ys  # (n_trajs, traj_len, y_dim)


# ---------------------------------------------------------------------------
# K-Means clustering of particles  (numpy port of get_multimodal_estimates)
# ---------------------------------------------------------------------------

def kmeans_particles(particles, n_modes, init_method, previous_centroids=None, n_iter=10):
    """
    Simple k-means on a set of particles (numpy).

    Parameters
    ----------
    particles          : (N, x_dim)
    n_modes            : int
    init_method        : 'random' or 'kmeanspp'
    previous_centroids : (n_modes, x_dim) or None
                         If provided, centroids are matched to previous ones via
                         the Hungarian algorithm to avoid label-switching.
    n_iter             : int  — k-means iterations.

    Returns
    -------
    centroids : (n_modes, x_dim)  — ordered to match previous_centroids
    labels    : (N,)              — cluster assignment for each particle
    """
    N, x_dim = particles.shape

    # --- 1. Initialise centroids (random subset of particles) ---
    if init_method == 'random':
        # --- 1.a. random subset of particles ---
        idx = np.random.choice(N, size=n_modes, replace=False)
        centroids = particles[idx].copy()
    else:
        # --- 1. with k-means++ (vectorized) ---
        idx = np.random.randint(N)
        centroids = [particles[idx].copy()]  # random particle used as first centroid
        for _ in range(n_modes - 1):
            # squared distance from each particle to its nearest centroid so far
            c_arr = np.array(centroids)  # (k, x_dim)
            diffs = particles[:, np.newaxis, :] - c_arr[np.newaxis, :, :]  # (N, k, x_dim)
            dists = np.linalg.norm(diffs, axis=-1).min(axis=1) ** 2  # (N,)
            probs = dists / dists.sum()
            idx = np.random.choice(N, p=probs)
            centroids.append(particles[idx].copy())
        centroids = np.array(centroids)  # (n_modes, x_dim)

    labels = np.zeros(N, dtype=int)
    for _ in range(n_iter):
        diffs = particles[:, np.newaxis, :] - centroids[np.newaxis, :, :]  # (N, K, x_dim)
        dists = np.linalg.norm(diffs, axis=-1)  # (N, K)
        labels = np.argmin(dists, axis=1)  # (N,)
        for k in range(n_modes):
            mask = labels == k
            if mask.any():
                centroids[k] = particles[mask].mean(axis=0)

    # --- 2. Tracking: match centroids to previous via Hungarian ---
    if previous_centroids is not None:
        cost = np.linalg.norm(
            previous_centroids[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=-1
        )  # (n_modes, n_modes)
        _, col_ind = linear_sum_assignment(cost)
        centroids = centroids[col_ind]
        remap = np.zeros(n_modes, dtype=int)
        remap[col_ind] = np.arange(n_modes)
        labels = remap[labels]
    # else:
    #     sort_idx = np.argsort(centroids[:, 0])
    #     centroids = centroids[sort_idx]
    #     remap = np.zeros(n_modes, dtype=int)
    #     remap[sort_idx] = np.arange(n_modes)
    #     labels = remap[labels]

    return centroids, labels


def _weighted_median_per_cluster(p_k, w_k, x_dim):
    """Weighted median of cluster particles, independently per dimension."""
    w_k = w_k / w_k.sum() if w_k.sum() > 0 else np.ones(len(w_k)) / len(w_k)
    median = np.zeros(x_dim)
    for d in range(x_dim):
        sort_idx = np.argsort(p_k[:, d])
        cumw = np.cumsum(w_k[sort_idx])
        cumw /= cumw[-1]
        idx = np.clip(np.searchsorted(cumw, 0.5), 0, len(p_k) - 1)
        median[d] = p_k[sort_idx[idx], d]
    return median


# ---------------------------------------------------------------------------
# Bootstrap Particle Filter
# ---------------------------------------------------------------------------

def particle_filter(ys_traj, n_particles, init_method, noise_std,
                    process_std, x0_low, x0_high, dt, get_derivs,
                    get_y, resample_method='systematic', n_modes=1):
    """
    Bootstrap SIR Particle Filter.

    Parameters
    ----------
    ys_traj          : (traj_len, y_dim)
    n_particles      : int
    init_method      : 'random' or 'kmeanspp'
    noise_std        : float
    process_std      : float
    x0_low, x0_high  : (x_dim,)
    dt               : float
    get_derivs       : callable(x, u=None) -> dx/dt
    get_y            : callable(x) -> y
    resample_method  : 'systematic' or 'multinomial'
    n_modes          : int — number of k-means modes.
                       1 = single weighted median (no clustering).
                       >1 = one weighted median per cluster.

    Returns
    -------
    particles_history  : (traj_len, n_particles, x_dim)
    weights_history    : (traj_len, n_particles)
    modal_estimates    : (n_modes, traj_len, x_dim)  If n_modes=1, shape is (1, traj_len, x_dim).
    resample_steps     : list of int
    labels_history     : (traj_len, n_particles)
    """
    traj_len, y_dim = ys_traj.shape
    x_dim = len(x0_low)
    meas_std = noise_std if noise_std > 1e-8 else 1e-2

    particles = np.random.uniform(x0_low, x0_high,
                                  size=(n_particles, x_dim)).astype(np.float32)
    weights = np.ones(n_particles, dtype=np.float32) / n_particles

    particles_history = np.zeros((traj_len, n_particles, x_dim), dtype=np.float32)
    weights_history = np.zeros((traj_len, n_particles), dtype=np.float32)
    labels_history = np.zeros((traj_len, n_particles), dtype=int)  # cluster label per particle
    modal_estimates = np.zeros((n_modes, traj_len, x_dim))
    resample_steps = []
    previous_centroids = None

    if resample_method == 'systematic':
        resample_fn = _systematic_resample
    else:
        resample_fn = _multinomial_resample

    for k in range(traj_len):
        y_k = ys_traj[k]

        # ----- 1. Propagate -----
        if k > 0:
            noise = np.random.normal(0, process_std, size=particles.shape).astype(np.float32) \
                if process_std > 1e-8 else 0.0
            particles = (RK4(get_derivs, dt, particles) + dt * noise).astype(np.float32)

        # ----- 2. Weighting -----
        y_pred = get_y(particles)
        diff = y_pred - y_k[np.newaxis, :]
        log_w = -0.5 * np.sum(diff ** 2, axis=-1) / meas_std ** 2
        log_w -= log_w.max()
        weights = np.exp(log_w)
        weights /= weights.sum()

        # ----- 3. Store -----
        particles_history[k] = particles
        weights_history[k] = weights

        # ----- 4. Modal estimate -----
        if n_modes == 1:
            modal_estimates[0, k] = _weighted_median_per_cluster(particles, weights, x_dim)
            labels_history[k] = 0  # all particles belong to mode 0
        else:
            centroids, labels = kmeans_particles(particles, n_modes, init_method,
                                                 previous_centroids=previous_centroids)
            previous_centroids = centroids
            labels_history[k] = labels
            for m in range(n_modes):
                mask = labels == m
                if mask.any():
                    modal_estimates[m, k] = _weighted_median_per_cluster(
                        particles[mask], weights[mask], x_dim)
                else:
                    print(f"ATTENTION! There were no particles for mode {m} at time step k={k} -- copying from k-1")
                    modal_estimates[m, k] = modal_estimates[m, k - 1] if k > 0 else 0.0

        # ----- 5. Resample if N_eff < 50% -----
        n_eff = 1.0 / np.sum(weights ** 2)
        if n_eff < n_particles / 2:
            particles = resample_fn(particles, weights)
            weights = np.ones(n_particles, dtype=np.float32) / n_particles
            resample_steps.append(k)

    return particles_history, weights_history, modal_estimates, resample_steps, labels_history


def _systematic_resample(particles, weights):
    """
    Systematic resampling, single random draw.
    Positions: (u + 0, u+1, ..., u+N-1) / N  with u ~ U[0,1).
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
    """
    N = len(weights)
    indices = np.random.choice(N, size=N, p=weights)
    return particles[indices]


# ---------------------------------------------------------------------------
# Set-valued estimation metrics (Hausdorff semi-distances)
# Only implemented for 'Test' system where X*(t) = {x(t), -x(t)}
# ---------------------------------------------------------------------------

def get_true_set(x_true):
    """
    Returns X*(t) for the Test system: {x(t), -x(t)}.
    x_true : (traj_len, x_dim)
    Returns : (traj_len, 2, x_dim)
    """
    assert args.dataset == 'Test', \
        "get_true_set is only implemented for the 'Test' dataset. " \
        "X*(t) for other systems must be defined separately."
    return np.stack([x_true, -x_true], axis=1)  # (traj_len, 2, x_dim)


def compute_hausdorff_metrics(modal_estimates, x_true):
    """
    Computes the two directed Modified Hausdorff semi-distances.

    Parameters
    ----------
    modal_estimates : (n_modes, traj_len, x_dim)   — X_hat(t)
    x_true          : (traj_len, x_dim)            — one ground-truth trajectory

    Returns
    -------
    precision_curve : (traj_len,)
        At each t: mean over x_hat in X_hat of  min_{x* in X*} ||x_hat - x*||
    coverage_curve  : (traj_len,)
        At each t: mean over x* in X* of  min_{x_hat in X_hat} ||x* - x_hat||
    """
    X_hat = modal_estimates.transpose(1, 0, 2)  # (traj_len, n_modes, x_dim)
    X_star = get_true_set(x_true)  # (traj_len, 2, x_dim)
    traj_len = x_true.shape[0]

    precision_curve = np.zeros(traj_len)
    coverage_curve = np.zeros(traj_len)

    for t in range(traj_len):
        xhat = X_hat[t]  # (n_modes, x_dim)
        xstar = X_star[t]  # (2, x_dim)

        # dist[i, j] = ||xhat_i - xstar_j||
        dist = np.linalg.norm(
            xhat[:, np.newaxis, :] - xstar[np.newaxis, :, :], axis=-1
        )  # (n_modes, 2)

        # Precision: for each x_hat, find closest x*. Then, mean over x_hat
        precision_curve[t] = dist.min(axis=1).mean()

        # Coverage: for each x*, find closest x_hat. Then, mean over x*
        coverage_curve[t] = dist.min(axis=0).mean()

    return precision_curve, coverage_curve


# ---------------------------------------------------------------------------
# Run filter on every trajectory
# ---------------------------------------------------------------------------
print(f"Running Bootstrap Particle Filter [N={args.n_particles}, "
      f"noise_std={args.noise_std}, process_std={args.process_std}]")

all_particles = []  # list of (traj_len, n_particles, x_dim)
all_weights = []  # list of (traj_len, n_particles)
all_modal_estimates = []  # list of (n_modes, traj_len, x_dim)
all_resample_steps = []  # list of lists of int
all_labels = []  # list of (traj_len, n_particles)

for i in range(args.n_trajs):
    print(f"  trajectory {i + 1:2d}/{args.n_trajs} ...", end=" ")
    ph, wh, me, rs, lh = particle_filter(ys_traj=ys[i],
                                         n_particles=args.n_particles,
                                         init_method=args.kmeans_init,
                                         noise_std=args.noise_std,
                                         process_std=args.pf_artificial_w_noise_std,
                                         x0_low=KKLDataset.x0_low,
                                         x0_high=KKLDataset.x0_high,
                                         dt=dt,
                                         get_derivs=KKLDataset.get_derivs,
                                         get_y=KKLDataset.get_y,
                                         resample_method=args.resample_method,
                                         n_modes=args.n_modes, )
    all_particles.append(ph)
    all_weights.append(wh)
    all_modal_estimates.append(me)
    all_resample_steps.append(rs)
    all_labels.append(lh)

    if args.dataset == 'Test':
        pc, cc = compute_hausdorff_metrics(me, xs[i])
        print(f"Precision={pc.mean():.4f}  Coverage={cc.mean():.4f}")
    else:
        # Oracle MSE: best-case error over modes
        err = ((me - xs[i][np.newaxis]) ** 2).mean(axis=-1)  # (n_modes, traj_len)
        mse_oracle = err.min(axis=0).mean()  # scalar
        print(f"Oracle MSE={mse_oracle:.4f}  RMSE={np.sqrt(mse_oracle):.4f}")

print("Done.\n\n\tWorking on plots and metrics ...")


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
    traj_len, n_particles, x_dim = particles.shape
    result = {p: np.zeros((traj_len, x_dim)) for p in percentiles}
    for t in range(traj_len):
        for d in range(x_dim):
            sorted_idx = np.argsort(particles[t, :, d])
            w_sorted = weights[t, sorted_idx]
            p_sorted = particles[t, sorted_idx, d]
            cumw = np.cumsum(w_sorted)
            cumw /= cumw[-1]  # normalise to exactly 1
            for p in percentiles:
                idx = np.searchsorted(cumw, p / 100.0)
                idx = np.clip(idx, 0, n_particles - 1)
                result[p][t, d] = p_sorted[idx]
    return result


def plot_probability_tube(b=0):
    sns.set(style="ticks", context="talk", font_scale=1.2)
    plt.rcParams["font.family"] = "serif"
    colors = sns.color_palette("Paired", n_colors=12)
    mode_colors = sns.color_palette("Set1", n_colors=max(args.n_modes, 2))

    t = ts[b]
    xb = xs[b]
    yb = ys[b]
    ph = all_particles[b]  # (traj_len, n_particles, x_dim)
    wh = all_weights[b]  # (traj_len, n_particles)
    me = all_modal_estimates[b]  # (n_modes, traj_len, x_dim)
    lh = all_labels[b]  # (traj_len, n_particles)
    resample_times = t[all_resample_steps[b]] if all_resample_steps[b] else []
    multimodal = args.n_modes > 1

    # percentiles per mode: dict {m: {pct: (traj_len, x_dim)}}
    pcts_per_mode = {}
    for m in range(args.n_modes):
        # build weights zeroed out for other clusters different from m, then renormalize
        wh_m = np.where(lh == m, wh, 0.0)  # (traj_len, n_particles)
        wh_m_sum = wh_m.sum(axis=1, keepdims=True)
        wh_m_sum = np.where(wh_m_sum > 0, wh_m_sum, 1.0)
        wh_m = wh_m / wh_m_sum
        pcts_per_mode[m] = compute_weighted_percentiles(ph, wh_m, percentiles=[5, 95])

    fig, axes = plt.subplots(x_dim + 1, 1,
                             figsize=(12, 3 * (x_dim + 1)),
                             sharex=True)

    # --- top row: measurement y ---
    ax = axes[0]
    for j in range(y_dim):
        ax.plot(t, yb[:, j], color=colors[3 + j], linewidth=1.2,
                label=f'$y_{j + 1}$' + (' (noisy)' if args.noise_std > 0 else ''))
    for rt in resample_times:
        ax.axvline(rt, color=colors[9], linewidth=1.5, alpha=0.4, label='resample' if rt == 0 else None)
    ax.set_ylabel('$y$')
    ax.legend(loc='right', fontsize=10)
    ax.grid(True, which='both', alpha=0.4)
    mode_str = f', k={args.n_modes} modes' if multimodal else ''
    noise_str = f'noise={args.noise_std}' if args.noise_std > 0 else 'noiseless'
    ax.set_title(f"Particle Filter  —  {args.dataset}  "
                 f"(N={args.n_particles}, {noise_str}, "
                 f"{len(resample_times)} resamplings{mode_str})", fontsize=18)

    # --- one row per state ---
    for i in range(x_dim):
        ax = axes[i + 1]

        # # scatter: particle cloud
        # t_expanded = np.repeat(t, args.n_particles)
        # p_expanded = ph[:, :, i].flatten()
        # ax.scatter(t_expanded, p_expanded, s=0.1, color='skyblue', alpha=0.05,
        #            label='Particles')

        # 90% tube per mode
        for m in range(args.n_modes):
            ax.fill_between(t,
                            pcts_per_mode[m][5][:, i],
                            pcts_per_mode[m][95][:, i],
                            alpha=0.20, color=mode_colors[m],
                            label='90%% interval' if not multimodal
                            else '90%% interval (mode %i)' % m)

        # modal estimates  — me has shape (n_modes, traj_len, x_dim)
        for m in range(args.n_modes):
            label = r'$\hat x_{%i}^{(%i)}$' % (i + 1, m) if multimodal \
                else r'$\hat x_{%i}$ (median)' % (i + 1)
            ax.plot(t, me[m, :, i],
                    color=mode_colors[m], linewidth=1.8, linestyle='--',
                    label=label)

        # ground truth
        ax.plot(t, xb[:, i], color=colors[3], linewidth=1.2,
                label='$x_{%i}$ (true)' % (i + 1))

        # resample instants
        for k, rt in enumerate(resample_times):
            ax.axvline(rt, color=colors[9], linewidth=1.5, alpha=0.4, )

        ax.set_ylabel(f'$x_{i + 1}$')
        ax.legend(loc='right', fontsize=9)
        ax.grid(True, which='both', alpha=0.4)

    axes[-1].set_xlabel('time (s)')
    plt.tight_layout()
    plt.savefig('particle_filter_tube.pdf', bbox_inches='tight', format='pdf')
    plt.savefig('particle_filter_tube.png', dpi=150, bbox_inches='tight')
    print("Saved: particle_filter_tube.png / .pdf")
    plt.show()

    # --- per-mode MSE for this trajectory (avg over time only) ---
    print(f"\n===== Modal MSE   —   trajectory {b} =====")
    for m in range(args.n_modes):
        mse_m = ((me[m] - xb) ** 2).mean()
        print(f"  Mode {m}: MSE = {mse_m:.5f}   RMSE = {np.sqrt(mse_m):.5f}")
    print("=" * 40)

    return fig


# ---------------------------------------------------------------------------
# Plot  — Modal MSE metrics
# ---------------------------------------------------------------------------

def plot_mse_over_time():
    """
    1) Best-mode oracle curve (per time steo):
       At each t, pick the mode k with the smallest squared error to the true state.
       Then, average over trajectories → one curve showing the oracle lower bound.

    2) Hausdorff precision & coverage curves (avg over trajs):
       Only for Test system. See paper for details on their definitions.
       Scalar summaries printed to console.
    """
    sns.set(style="ticks", context="talk", font_scale=1.2)
    plt.rcParams["font.family"] = "serif"
    colors = sns.color_palette("Paired", n_colors=10)

    t = ts[0]

    # --- best-mode oracle ---
    err_modal_all = []
    err_best_all = []
    for i in range(args.n_trajs):
        me = all_modal_estimates[i]  # (n_modes, traj_len, x_dim)
        x_true = xs[i]
        err = ((me - x_true[np.newaxis]) ** 2).mean(axis=-1)  # (n_modes, traj_len)
        err_modal_all.append(err)
        err_best_all.append(err.min(axis=0))

    err_modal_all = np.array(err_modal_all)  # (n_trajs, n_modes, traj_len)
    err_best_all = np.array(err_best_all)  # (n_trajs, traj_len)
    best_curve = err_best_all.mean(axis=0)  # (traj_len,)

    print("\n===== Oracle MSE (best mode, avg over time & trajectories) =====")
    print(f"  MSE = {best_curve.mean():.5f}   RMSE = {np.sqrt(best_curve.mean()):.5f}")
    print("=" * 64)

    # --- Hausdorff metrics (Test only) ---
    hausdorff_available = (args.dataset == 'Test')
    if hausdorff_available:
        prec_all = []
        cov_all = []
        for i in range(args.n_trajs):
            pc, cc = compute_hausdorff_metrics(all_modal_estimates[i], xs[i])
            prec_all.append(pc)
            cov_all.append(cc)
        prec_curve = np.array(prec_all).mean(axis=0)  # (traj_len,)
        cov_curve = np.array(cov_all).mean(axis=0)

        print("\n===== Hausdorff Metrics (avg over trajectories) =====")
        print(f"  Precision  (X_hat -> X*): {prec_curve.mean():.5f}")
        print(f"  Coverage   (X* -> X_hat): {cov_curve.mean():.5f}")
        print("=" * 50)

    # --- plot ---
    n_subplots = 2 if hausdorff_available else 1
    fig, axes = plt.subplots(n_subplots, 1, figsize=(10, 4 * n_subplots), sharex=True)
    if n_subplots == 1:
        axes = [axes]

    # subplot 1: oracle MSE
    axes[0].semilogy(t, best_curve, color=colors[9], linewidth=1.5,
                     label=r'Error of best mode (MSE=%.3e)' % best_curve.mean())
    axes[0].set_ylabel('MSE')
    axes[0].set_title(f'Particle Filter  —  {args.dataset}  '
                      f'(k={args.n_modes}, N={args.n_particles})')
    axes[0].legend(loc='upper right', fontsize=11)
    axes[0].grid(True, which='both', alpha=0.4)

    # subplot 2: Hausdorff precision & coverage
    if hausdorff_available:
        axes[1].plot(t, prec_curve, color=colors[1], linewidth=1.5,
                     label=r'Precision $\bar{d}MHD (\hat{\mathcal{X}} , \mathcal{X}^{*})$'
                           r'  (%.3e)' % prec_curve.mean())
        axes[1].plot(t, cov_curve, color=colors[3], linewidth=1.5, linestyle='--',
                     label=r'Coverage $\bar{d}MHD (\mathcal{X}^{*}, \hat{\mathcal{X}})$'
                           r'  (%.3e)' % cov_curve.mean())
        axes[1].set_ylabel('Distance')
        axes[1].legend(loc='upper right', fontsize=11)
        axes[1].grid(True, which='both', alpha=0.4)

    axes[-1].set_xlabel('time (s)')
    plt.tight_layout()
    plt.savefig('particle_filter_mse.pdf', bbox_inches='tight', format='pdf')
    plt.savefig('particle_filter_mse.png', dpi=150, bbox_inches='tight')
    print("Saved: particle_filter_mse.png / .pdf")
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Debug: scatter plot in x1-x2 space, one frame per timestep
# ---------------------------------------------------------------------------

def plot_particle_scatter_frames(b=0, folder='debug_scatter'):
    """
    For each timestep, save a scatter plot of particles in (x1, x2) space,
    colored by k-means cluster, with centroids and ground truth marked.
    All frames saved to `folder/` for visual debugging.
    """
    import os
    os.makedirs(folder, exist_ok=True)

    sns.set(style="ticks", context="talk", font_scale=1.0)
    plt.rcParams["font.family"] = "serif"
    mode_colors = sns.color_palette("Set1", n_colors=max(args.n_modes, 2))

    ph = all_particles[b]  # (traj_len, N, x_dim)
    wh = all_weights[b]  # (traj_len, N)
    lh = all_labels[b]  # (traj_len, N)
    me = all_modal_estimates[b]  # (n_modes, traj_len, x_dim)
    xb = xs[b]  # (traj_len, x_dim)
    t = ts[b]  # (traj_len,)
    rs = set(all_resample_steps[b])

    # axis limits: fixed across all frames for comparability
    x1_min, x1_max = KKLDataset.x0_low[0] * 1.3, KKLDataset.x0_high[0] * 1.3
    x2_min, x2_max = KKLDataset.x0_low[1] * 1.3, KKLDataset.x0_high[1] * 1.3

    traj_len = ph.shape[0]
    print(f"Saving {traj_len} scatter frames to '{folder}/' ...")

    for k in range(traj_len):
        fig, ax = plt.subplots(figsize=(5, 5))

        # --- particles colored by cluster ---
        for m in range(args.n_modes):
            mask = lh[k] == m
            ax.scatter(ph[k, mask, 0], ph[k, mask, 1],
                       s=2, alpha=0.3, color=mode_colors[m],
                       label=f'Mode {m} ({mask.sum()} particles)')

        # --- centroids ---
        for m in range(args.n_modes):
            ax.scatter(me[m, k, 0], me[m, k, 1],
                       s=60, alpha=0.2, marker='X', color=mode_colors[m],
                       edgecolors='black', linewidths=0.8, zorder=5,
                       label=f'Centroid {m}')

        # # --- ground truth x and its symmetric -x ---
        # ax.scatter(xb[k, 0], xb[k, 1],
        #            s=150, marker='*', color='black', zorder=6, label='$x$ (true)')
        # ax.scatter(-xb[k, 0], -xb[k, 1],
        #            s=150, marker='*', color='gray', zorder=6, label='$-x$ (symmetric)')

        # --- resampling indicator ---
        resampled_str = ' [RSPL]' if k in rs else ''

        ax.set_xlim(x1_min, x1_max)
        ax.set_ylim(x2_min, x2_max)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_title(f't = {t[k]:.3f}{resampled_str}  (step {k}/{traj_len - 1})')
        ax.legend(loc='upper right', fontsize=7, markerscale=1.5)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        fname = os.path.join(folder, f'frame_{k:04d}.png')
        plt.savefig(fname, dpi=80, bbox_inches='tight')
        plt.close(fig)

        if k % 50 == 0:
            print(f"  {k}/{traj_len} frames saved...")

    print(f"Done. All frames saved to '{folder}/'")
    print(f"Tip: convert to video with:  ffmpeg -r 20 -i {folder}/frame_%04d.png -vcodec libx264 debug.mp4")


# ---------------------------------------------------------------------------
# Run plots
# ---------------------------------------------------------------------------
plot_probability_tube(b=args.batch)
plot_probability_tube(b=1)
plot_probability_tube(b=2)
plot_probability_tube(b=3)
plot_probability_tube(b=4)
plot_mse_over_time()
plot_particle_scatter_frames(b=4)
