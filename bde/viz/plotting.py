"""This here has some plotters!"""

import matplotlib.pyplot as plt
import os

import numpy as np
import os
import matplotlib.pyplot as plt


def plot_pred_vs_true(y_pred, y_true, y_pred_err, title, savepath):
    """
    #TODO: documentation
    Parameters
    ----------
    y_pred
    y_true
    y_pred_err
    title
    savepath

    Returns
    -------

    """

    def _to_1d(a):
        a = np.asarray(a)  # handles JAX DeviceArray too
        a = np.squeeze(a)  # drop singleton dims
        return a.reshape(-1)  # ensure 1D

    y_true_1d = _to_1d(y_true)
    y_pred_1d = _to_1d(y_pred)

    yerr = None
    if y_pred_err is not None:
        err = np.asarray(y_pred_err)
        # if it's (N,1) or (N,), squeeze to 1D
        err = np.squeeze(err)
        # if it looks like variance (non-negative), take sqrt to get std
        if np.all(err >= 0):
            err = np.sqrt(err)
        yerr = err.reshape(-1)

    fig, ax = plt.subplots()
    ax.errorbar(y_true_1d, y_pred_1d, yerr=yerr, fmt='o', color="#539ecd", alpha=0.5)

    lim_min = float(min(y_true_1d.min(), y_pred_1d.min()))
    lim_max = float(max(y_true_1d.max(), y_pred_1d.max()))
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", linewidth=1, label="y = x")

    ax.set_xlabel("True values")
    ax.set_ylabel("Predicted values")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    if savepath is not None:
        os.makedirs(savepath, exist_ok=True)
        plt.savefig(os.path.join(savepath, f"{title}.png"), bbox_inches="tight")
    plt.close(fig)
