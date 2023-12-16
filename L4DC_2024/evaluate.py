import numpy as np
from matplotlib import pyplot as plt
import os


from train import Problem


FIGURES_DIR = "figs"


def get_result_name(pb: Problem, use_encoder: bool = None):
    if use_encoder is None:
        model_name = "mixed"
    elif use_encoder:
        model_name = "autoencoder"
    else:
        model_name = "decoder"
    return "%s_%s_noise%s" % (model_name, pb.name, pb.noise_std)


def get_figure_path(pb: Problem, use_encoder, extension=".png"):
    name = get_result_name(pb, use_encoder)
    return os.path.join(FIGURES_DIR,  name + extension)


def eval_errors(xs, xs_predict, transient_len):
    errs = np.sqrt(((xs - xs_predict.numpy()) ** 2).mean(-1).mean(0))
    tot = errs.mean()
    transient = errs[:transient_len].mean()
    asymptotic = errs[transient_len:].mean()
    print("errs: total %f transient %f asymptotic %f" %
          (tot, transient, asymptotic))


def render_rossler(pb, ts, ys, xs, xs_predict, idx, transient_len, use_encoder):
    eval_errors(xs, xs_predict, transient_len)

    # -- plot
    fig = plt.figure()

    plt.subplot(311)
    plt.plot(ts, xs[idx, :, 0], 'b-',
             ts, xs_predict[idx, :, 0], 'r--')
    plt.legend(['$x_1$', '$\hat{x}_1$'], loc=1)

    plt.subplot(312)
    plt.plot(ts, ys[idx, :, 0], 'steelblue',
             ts, xs[idx, :, 1], 'b-',
             ts, xs_predict[idx, :, 1], 'r--')
    plt.legend(['$y$', '$x_2$', '$\hat{x}_2$'], loc=1)

    plt.subplot(313)
    plt.plot(ts, xs[idx, :, 2], 'b-',
             ts, xs_predict[idx, :, 2], 'r--')
    plt.legend(['$x_3$', '$\hat{x}_3$'], loc=1)

    plt.xlabel('time')

    # -- save figure
    fig_path = get_figure_path(pb, use_encoder, extension=".pdf")
    fig.savefig(fig_path, bbox_inches='tight')
