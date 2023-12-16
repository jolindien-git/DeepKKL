import torch
import numpy as np
import matplotlib.pyplot as plt

from dataset import VDP_Dataset as KKLDataset


test_dataset = KKLDataset(n_samples=int(1e5), std_x=1)
dt = test_dataset.dt
x_dim = test_dataset.x_dim


def observer_withControl(model, simu_len, noise_std=0):
    z_dim = model.z_dim
    X, Y, U = test_dataset.generate_sequences(n_simus=1,
                                              simus_len=simu_len,
                                              autonomous=False,
                                              noise_std=noise_std)

    y_seq = Y[0, :, :]
    u_seq = U[0, :, :]

    std_x = model.std_x

    y_seq = torch.tensor(y_seq).unsqueeze(0)
    x_hat = torch.zeros((1, simu_len, x_dim))
    z_hat = torch.zeros((1, simu_len, z_dim))
    with torch.no_grad():
        z_hat[:, 0, :] = model.encode(x_hat[:, 0, :] / std_x)

    for k in range(simu_len):
        if k == 0:
            z_next = z_hat[:, k, :]
        else:
            z = z_hat[:, k - 1, :]
            z_next = model.z_next(z, y_seq[:, k - 1, :])
            u = torch.tensor(u_seq[k - 1, :], dtype=torch.float)
            x_hat_ = x_hat[:, k - 1, :].detach().numpy()
            x_hat_next_u = test_dataset.get_x_next(x_hat_,
                                                   u.unsqueeze(0).numpy())
            x_hat_next = test_dataset.get_x_next(x_hat_)
            z_next += model.encode(torch.tensor(x_hat_next_u / std_x)) \
                - model.encode(torch.tensor(x_hat_next / std_x))
            # dT_dx = 1 / std_x * torch.autograd.functional.jacobian(model.T,
            #                                            x_hat[:, k - 1, :].squeeze(0) / std_x)
            # z_next += dt * dT_dx[:, 1] * u_seq[k - 1, :]

        z_hat[:, k, :] = z_next
        x_hat[:, k, :] = model.decode(z_next) * std_x

    x_hat = x_hat.squeeze(0).detach().numpy()
    z_hat = z_hat.squeeze(0).detach().numpy()

    # plots
    plt.figure()
    t = [k * dt for k in range(simu_len)]

    # plot x and obs
    x_seq = X[0, :, :]
    plt.subplot(411)
    legend = []
    for i in range(x_dim):
        plt.plot(t, x_seq[:, i])
        legend.append('$x_%i$' % (i+1,))
    for i in range(x_dim):
        plt.plot(t, x_hat[:, i], '--')
        legend.append('$\hat{x}_%i$' % (i+1,))
    plt.legend(legend)

    # noise
    plt.subplot(412)
    noise = np.float32(np.random.normal(0, noise_std, size=(len(t),)))
    plt.plot(t, noise)
    plt.legend(['additive noise on y'])

    # plot u
    plt.subplot(413)
    plt.plot(t, u_seq[:, 0])
    plt.legend(['$u$'])

    # plot errors
    err = ((x_seq - x_hat)**2).mean(axis=1)
    plt.subplot(414)
    plt.plot(t, err)
    plt.legend(['$||x - \hat{x}||_{MSE}$'])

    return x_seq, x_hat, u_seq, noise
