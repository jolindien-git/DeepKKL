import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt

from dataset import Example_Dataset as KKLDataset
from NN_models import KKL_Autoencoder, kkl_pde_residuals


LAMBDAS = [-1.0, -1.5, -2.0, -2.5]  # observer eigenvalues
CRITERION = torch.nn.MSELoss()


# %% --- init datasets ---
BATCH_SIZE = 100

train_dataset = KKLDataset(use_traj=False,
                           n_samples=10000 * 10,
                           traj_len=-1,
                           noise_std=0)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

valid_dataset = KKLDataset(use_traj=False,
                           n_samples=10000,
                           traj_len=-1,
                           noise_std=0)
valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset))


# %% --- init model ---
model = KKL_Autoencoder(x_dim=KKLDataset.x_dim,
                        y_dim=KKLDataset.y_dim,
                        lambdas=LAMBDAS,
                        net_arch=[128*2, 128*2])


# %% --- training T ---
LEARNING_RATE = 1e-4
EPOCHS = 100

optimizer_T = torch.optim.Adam(model.T.parameters(), lr=LEARNING_RATE)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer_T, T_max=EPOCHS)

print("==== training T (encoder) ====")
for epoch in range(EPOCHS):
    train_losses = []
    
    #-- train
    for x, x_dot, y in train_loader:        
        residuals = kkl_pde_residuals(x, x_dot, y, model)
        loss_T = CRITERION(residuals, torch.zeros_like(residuals))

        optimizer_T.zero_grad()
        loss_T.backward()
        optimizer_T.step()
        train_losses.append(loss_T.item())
    
    #-- valid
    with torch.no_grad():
        for i, (x, x_dot, y) in enumerate(valid_loader):
            assert(i == 0)
            residuals = kkl_pde_residuals(x, x_dot, y, model)
            valid_loss_T = CRITERION(residuals, torch.zeros_like(residuals))
    
    #-- verbose
    lr = scheduler.get_last_lr()[0]
    print('epoch %i train %.2e valid %.2e  lr %.2e' % 
          (epoch, np.mean(train_losses), valid_loss_T, lr))
    
    #-- update learning rate
    scheduler.step()


# %% --- training T⁻1 ---
LEARNING_RATE = 1e-4*10
EPOCHS = 100

optimizer_invT = torch.optim.Adam(model.invT.parameters(), lr=LEARNING_RATE)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer_invT, T_max=EPOCHS)

print("==== training pseudo inverse (decoder) ====")
for epoch in range(EPOCHS):
    train_losses = []

    for x, _, _ in train_loader:
        with torch.no_grad():
            z = model.encode(x)
        x_pred = model.decode(z)
        
        optimizer_invT.zero_grad()
        loss_invT = CRITERION(x, x_pred)
        loss_invT.backward()
        optimizer_invT.step()

        train_losses.append(loss_invT.item())

    with torch.no_grad():
        for i, (x, _, _) in enumerate(valid_loader):
            assert(i == 0)
            z = model.encode(x)
            x_pred = model.decode(z)
            valid_loss_invT = CRITERION(x, x_pred)
    
    lr = scheduler.get_last_lr()[0]
    print('epoch %i train %.2e valid %.2e  lr %.2e' % 
          (epoch, np.mean(train_losses), valid_loss_invT, lr))
    
    scheduler.step()
    

# %% --- test the observer ---

ts, xs, ys, _ = valid_dataset.generate_trajectories(n_traj=1,
                                                    traj_len=1000,
                                                    noise_std=0)
ts, xs, ys = ts[0], xs[0], ys[0] # unsqueeze batch dim

tensor = lambda a : torch.tensor(a, dtype=torch.float32)

xs_obs, zs = np.zeros_like(xs), np.zeros((len(ts), model.z_dim))
for k, t in enumerate(ts):
    y = tensor(ys[k])
    if k == 0:
        z = tensor(zs[k])
    else:
        z = model.z_next(z, y, valid_dataset.dt)
    x_obs = model.decode(z)
    xs_obs[k] = x_obs.detach().numpy()


for i in range(model.x_dim):
    plt.subplot(model.x_dim, 1, i + 1)
    plt.plot(ts, xs[:, i], ts, xs_obs[:, i])
    plt.legend(['x_%i' % (i + 1), 'obs'])
    
    