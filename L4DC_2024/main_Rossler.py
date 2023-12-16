"""
Created: November 2023
Modified:
@author: Johan Peralez
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import os

from dataset import Rossler_Dataset
from NN_models import KKL_AutoEncoder, render_eigenvalues
from train import Problem, Algo, get_data, get_problem_data, train_autoencoder
from evaluate import eval_errors, get_result_name, render_rossler


# %% -- user parameters
MODELS_DIR = "models"

# -- problem definition
PB = Problem(dataset=Rossler_Dataset,
             noise_std=0,
             data_traj_number=1 * int(1e3),
             data_traj_len=1000,
             name="Rossler")

# -- algorithm options
ALGO = Algo(A_diag=False,
            z_dim=2*PB.dataset.x_dim + 1,
            batch_size=32,
            net_arch=[128, 128],
            epochs=100,
            lr_init=1e-2/2,
            criterion=torch.nn.HuberLoss(delta=1))

TRAIN_PERCENT = 90  # split data into train / valid


# %% --- init datasets ---
train_dataset, valid_dataset = get_problem_data(PB, TRAIN_PERCENT)
train_loader = DataLoader(train_dataset, batch_size=ALGO.batch_size,
                          shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset))


# %% --- init models ---
def get_model_init(algo=ALGO, pb=PB, use_encoder=False):
    return KKL_AutoEncoder(x_dim=pb.dataset.x_dim,
                           y_dim=pb.dataset.y_dim,
                           z_dim=algo.z_dim,
                           net_arch=algo.net_arch,
                           A_diag=algo.A_diag,
                           use_encoder=use_encoder)





def get_model_path(pb, use_encoder):
    return os.path.join(MODELS_DIR, get_result_name(pb, use_encoder) + ".pt")



def save_model(algo, pb, model):
    path = get_model_path(pb, model.use_encoder)
    print("saving", path)
    torch.save({
        'problem_dict': pb.__dict__,
        'algo_dict': algo.__dict__,
        'model_state_dict': model.state_dict(),
        'use_encoder': model.use_encoder
        },
        path)


def load_model(path):
    checkpoint = torch.load(path)
    algo = Algo(**checkpoint['algo_dict'])
    pb = Problem(**checkpoint['problem_dict'])
    model = get_model_init(algo, pb, checkpoint['use_encoder'])
    model.load_state_dict(checkpoint['model_state_dict'])
    return algo, pb, model


# %% --- transient observer: training \Tau (extension of T^{-1}) ---
decoder = get_model_init()
train_autoencoder(model=decoder,
                  train_loader=train_loader,
                  valid_loader=valid_loader,
                  algo=ALGO)
save_model(ALGO, PB, decoder)

# -- verbose: eigen values of A
render_eigenvalues(decoder.A)


# %% --- asymptotic observer: training T and T^{-1}

# -- init autoencoder with decoder
decoder_path = get_model_path(PB, use_encoder=False)
algo, pb, autoencoder = load_model(decoder_path)
autoencoder.use_encoder = True

if PB.noise_std == 0:
    autoencoder.A_frozen = True
train_autoencoder(model=autoencoder,
                  train_loader=train_loader,
                  valid_loader=valid_loader,
                  algo=ALGO)
save_model(ALGO, PB, autoencoder)

# -- verbose: eigen values of A
render_eigenvalues(autoencoder.A)


# %% -- evaluate (on a batch of simu)
TRAJ_LEN=1000
TRAJ_NUMBER=100
TRANSIENT_LEN=100

test_dataset = get_data(PB.dataset, TRAJ_NUMBER, TRAJ_LEN, PB.noise_std)
xs, ys = test_dataset[:]
ts = np.array(range(TRAJ_LEN)) * PB.dataset.dt
idx = np.random.randint(0, len(xs))  # pick a trajectory

# -- first observer: \hat{x} = \Tau(z)
algo, pb, decoder = load_model(get_model_path(PB, use_encoder=False))
with torch.no_grad():
    zs_obs1, xs_obs1 = decoder.trajectories(torch.tensor(ys))
render_rossler(PB, ts, ys, xs, xs_obs1, idx, TRANSIENT_LEN, False)
# -- second observer: \hat{x} = T^{-1}(z)
algo, pb, autoencoder = load_model(get_model_path(PB, use_encoder=True))
with torch.no_grad():
    zs_obs2, xs_obs2 = autoencoder.trajectories(torch.tensor(ys))
render_rossler(PB, ts, ys, xs, xs_obs2, idx, TRANSIENT_LEN, True)
# -- final observer:
with torch.no_grad():
    # -- 1) transient phase: \hat{x} = \Tau(z)
    xs_obs = xs_obs1.clone()
    # -- 2) switch: z = T(\hat{x})
    x_obs = xs_obs1[:, TRANSIENT_LEN - 1, :]
    # -- 3) asymptotic phase: \hat{x} = T^{-1}(z)
    ys2 = ys[:, TRANSIENT_LEN:, :]
    _, xs_obs[:, TRANSIENT_LEN:, :] = autoencoder.trajectories(
        torch.tensor(ys2), x_obs)
render_rossler(PB, ts, ys, xs, xs_obs, idx, TRANSIENT_LEN, None)


# %% switch
h_Rossler = lambda xs: xs[:, :, 1 : 2]
z_next = lambda z, y: decoder.z_next(z,y)

# -- compute the observers correction C:
# --    C = invT(Az + B y) - invT(Az + B y_obs)
C = [None] * 2
errs = [None] * 2
for i, (obs, xs_obs, zs) in enumerate([(decoder, xs_obs1, zs_obs1),
                                       (autoencoder, xs_obs2, zs_obs2)]):
    ys_obs = h_Rossler(xs_obs)
    with torch.no_grad():
        C[i] = obs.decode(z_next(zs, torch.Tensor(ys))) \
                - obs.decode(z_next(zs, torch.Tensor(ys_obs)))
    errs[i] = np.sqrt(((xs - xs_obs.numpy()) ** 2).mean(-1).mean(0))


# -- switching criteria on v
# --    v_next = a v + C' P C
# --    with 0 < a < 1, C' = transpose(C), P definite positive
a = .99
x_dim = PB.dataset.x_dim
P = torch.nn.Linear(x_dim, x_dim, bias=False)
P.weight.data = torch.diag(torch.Tensor([1] * x_dim))
vs = [torch.zeros(TRAJ_NUMBER, TRAJ_LEN),
      torch.zeros(TRAJ_NUMBER, TRAJ_LEN)]
for i in range(2):
    v = 0
    for k in range(TRAJ_LEN):
        v = a * v + (C[i][:, k, :] ** 2).sum(-1)
        vs[i][:, k] = v

# -- final observer:
o1 = xs_obs1.clone()
o1[vs[0] > vs[1]] = 0
o2 = xs_obs2.clone()
o2[vs[0] <= vs[1]] = 0
xs_obs = o1 + o2
render_rossler(PB, ts, ys, xs, xs_obs, idx, TRANSIENT_LEN, None)

# -- plot
fig = plt.figure()

plt.subplot(611)
plt.plot(C[0][idx, :, 0], '-',
         C[1][idx, :, 0], '--')#, label='C_0')
plt.legend(loc=1)

plt.subplot(612)
plt.plot(C[0][idx, :, 1], '-',
         C[1][idx, :, 1], '--')#, label='C_1')
plt.legend(loc=1)

plt.subplot(613)
plt.plot(C[0][idx, :, 2], '-',
         C[1][idx, :, 2], '--')#, label='C_2')
plt.legend(loc=1)

plt.subplot(614)
plt.plot(errs[0], '-',
         errs[1], '--', label='RMSE')
plt.legend(loc=1)

plt.subplot(615)
plt.plot(vs[0][idx], '-',
         vs[1][idx], '--', label='v')
plt.legend(loc=1)

plt.subplot(616)
test = torch.ones_like(o1)
test[vs[0] > vs[1]] = 0
plt.plot(test[idx, :, 0])
