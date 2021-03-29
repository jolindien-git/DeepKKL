import torch
import numpy as np
from torch.utils.data import DataLoader
import copy

from dataset import VDP_Dataset as KKLDataset
from NN_models import KKL_Autoencoder
import evaluate


LAMBDAS = [-.5, -1, -1.5]  # observer eigenvalues

BATCH_SIZE = 100
NET_ARCH = [500, 500]

N_SAMPLES_TRAIN = 200 * int(1e3)
N_SAMPLES_VALID = 50 * int(1e3)
CRITERION = torch.nn.MSELoss()


# %% --- init datasets ---
train_dataset = KKLDataset(n_samples=N_SAMPLES_TRAIN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

std_x = train_dataset.std_x

valid_dataset = KKLDataset(n_samples=N_SAMPLES_VALID, std_x=std_x)
valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset))

x_dim = train_dataset.x_dim
y_dim = train_dataset.y_dim
dt = train_dataset.dt


# %% --- init model ---
model = KKL_Autoencoder(x_dim=x_dim,
                        y_dim=y_dim,
                        dt=dt,
                        lambdas=LAMBDAS,
                        net_arch=NET_ARCH)

model.std_x = std_x


# %% --- training T ---
LEARNING_RATE = 1e-4
optimizer_T = torch.optim.Adam(model.T.parameters(), lr=LEARNING_RATE)

print("==== training T (encoder) ====")
best_loss = np.inf
for epoch in range(160):
    train_losses = []

    if epoch < 10:
        # scale z thanks to B
        loader = DataLoader(train_dataset, batch_size=len(train_dataset))
        scaling_data = list(loader)[0]
        with torch.no_grad():
            x = scaling_data[0]
            z = model.encode(x)
            scale = 1 / z.std(dim=0)
            for i, s in enumerate(scale):
                model.B.data[i] *= s
            print('\t B : ', *model.B.detach())

    for x, y, x_next, y_next in train_loader:
        optimizer_T.zero_grad()

        z = model.encode(x)
        z_dyn = model.z_next(z, y)
        z_next = model.encode(x_next)

        loss_T = CRITERION(z_next, z_dyn)

        loss_T.backward()
        optimizer_T.step()
        train_losses.append(loss_T.item())

    with torch.no_grad():
        for i, (x, y, x_next, y_next) in enumerate(valid_loader):
            assert(i == 0)
            z = model.encode(x)
            z_dyn = model.z_next(z, y)
            valid_loss_T = CRITERION(model.encode(x_next), z_dyn).item()

        if valid_loss_T < best_loss:
            best_loss = valid_loss_T
            best_model = copy.deepcopy(model)

    print('epoch %i train %.2e    valid %.2e' % (epoch, np.mean(train_losses),
                                                 valid_loss_T))

    if epoch > 0 and epoch % 50 == 0:
        optimizer_T.param_groups[0]['lr'] /= 4
        print('reduce optim lr = %.2e' % (optimizer_T.param_groups[0]['lr'],))

model = copy.deepcopy(best_model)


# %% --- training T⁻1 ---
LEARNING_RATE = 1e-4
optimizer_invT = torch.optim.Adam(model.Psi.parameters(), lr=LEARNING_RATE)

print("==== training pseudo inverse (decoder) ====")
best_loss = np.inf
for epoch in range(160):
    train_losses = []

    for x, y, x_next, y_next in train_loader:
        with torch.no_grad():
            z = model.encode(x)
            z_next = model.encode(x_next)
        optimizer_invT.zero_grad()
        loss_invT = CRITERION(x, model.decode(z)) + CRITERION(x_next, model.decode(z_next))
        loss_invT.backward()
        optimizer_invT.step()

        train_losses.append(loss_invT.item())

    with torch.no_grad():
        for i, (x, y, x_next, y_next) in enumerate(valid_loader):
            assert(i == 0)
            z = model.encode(x)
            valid_loss_invT = CRITERION(x, model.decode(z.detach())).item()

        if valid_loss_invT < best_loss:
            best_loss = valid_loss_invT
            best_model = copy.deepcopy(model)

    print('epoch %i train %.2e    valid %.2e' % (epoch, np.mean(train_losses),
                                                 valid_loss_invT))

    if epoch > 0 and epoch % 50 == 0:
        optimizer_invT.param_groups[0]['lr'] /= 4
        print('reduce optim lr = %.2e' % (optimizer_invT.param_groups[0]['lr'],))


# %% --- evaluate ---
x_seq, x_hat, u_seq, noise = evaluate.observer_withControl(best_model,
                                                           simu_len=4000,
                                                           noise_std=0.2)
