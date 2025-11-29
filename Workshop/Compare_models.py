import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from scipy.optimize import curve_fit
import pickle
import torch

def compute_sigma_over_mu(y_true, y_pred, bin_edges=None, eps=1e-8, min_bin_size=10):
    """
    Compute sigma/mu and its uncertainty for given predictions.

    Parameters
    ----------
    y_true : np.ndarray or torch.Tensor
        True target values.
    y_pred : np.ndarray or torch.Tensor
        Model predictions.
    bin_edges : np.ndarray, optional
        Bin edges for grouping by y_true. If None, default edges are used.
    eps : float
        Small constant to avoid division by zero.
    min_bin_size : int
        Minimum number of samples per bin to fit.

    Returns
    -------
    bin_centers : np.ndarray
    sigma_over_mu : np.ndarray
    sigma_over_mu_err : np.ndarray
    mu_vals : np.ndarray
    sigma_vals : np.ndarray
    mu_errs : np.ndarray
    sigma_errs : np.ndarray
    """
    # Convert torch tensors to numpy
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # Default bin edges if not provided
    if bin_edges is None:
        bin_edges = np.concatenate([
            np.arange(20, 120, 10),               # finer bins
            np.arange(120, y_true.max() + 20, 15) # coarser bins
        ])
        bin_edges = bin_edges[:-2]  # match your original code

    # Bin indices
    bin_indices = np.digitize(y_true, bin_edges)

    # Fit helper functions
    def truncated_gaussian(x, mu, sigma, a=0.9, b=1.1):
        a_, b_ = (a - mu) / sigma, (b - mu) / sigma
        return truncnorm.pdf(x, a_, b_, loc=mu, scale=sigma)

    def fit_func(x, mu, sigma):
        return truncated_gaussian(x, mu, sigma)

    bin_centers = []
    mu_vals, sigma_vals, mu_errs, sigma_errs = [], [], [], []

    # Loop over bins
    for i in range(1, len(bin_edges)):
        idx = np.where(bin_indices == i)[0]
        if len(idx) < min_bin_size:
            continue

        y_true_bin = y_true[idx]
        y_pred_bin = y_pred[idx]
        response = y_true_bin / (y_pred_bin + eps)

        # Histogram for fitting
        hist_vals, hist_edges = np.histogram(response, bins=50, range=(0.9, 1.1), density=True)
        hist_centers = 0.5 * (hist_edges[1:] + hist_edges[:-1])

        mu_guess, sigma_guess = np.mean(response), np.std(response)

        try:
            popt, pcov = curve_fit(fit_func, hist_centers, hist_vals, p0=[mu_guess, sigma_guess])
            mu_fit, sigma_fit = popt
            mu_err, sigma_err = np.sqrt(np.diag(pcov))

            mu_vals.append(mu_fit)
            sigma_vals.append(sigma_fit)
            mu_errs.append(mu_err)
            sigma_errs.append(sigma_err)
            bin_centers.append((bin_edges[i-1] + bin_edges[i]) / 2)
        except RuntimeError:
            # Fit failed, skip bin
            continue

    mu_vals = np.array(mu_vals)
    sigma_vals = np.array(sigma_vals)
    mu_errs = np.array(mu_errs)
    sigma_errs = np.array(sigma_errs)
    bin_centers = np.array(bin_centers)

    sigma_over_mu = sigma_vals / mu_vals
    sigma_over_mu_err = sigma_over_mu * np.sqrt((sigma_errs / sigma_vals)**2 +
                                                (mu_errs / mu_vals)**2)

    return bin_centers, sigma_over_mu, sigma_over_mu_err, mu_vals, sigma_vals, mu_errs, sigma_errs


def fit_resolution(bin_centers, sigma_over_mu, sigma_over_mu_err):
    def resolution_model(E, S, C):
        return np.sqrt((S / np.sqrt(E))**2 + C**2)

    popt, pcov = curve_fit(
        resolution_model, bin_centers, sigma_over_mu,
        sigma=sigma_over_mu_err, p0=[0.3, 0.01], absolute_sigma=True
    )
    S_fit, C_fit = popt
    S_err, C_err = np.sqrt(np.diag(pcov))
    return (S_fit, S_err, C_fit, C_err), resolution_model



def plot_models(models_data, bin_edges, ref_idx=0):
    """
    models_data: list of tuples -> [(y_true, y_pred, label, color), ...]
    ref_idx: index of reference model for ratio plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True, 
                                   gridspec_kw={'height_ratios': [3, 1]})

    fits = []
    x_plot = np.linspace(min(bin_edges), max(bin_edges), 300)

    for (y_true, y_pred, label, color) in models_data:
        bc, so_mu, so_mu_err, _, _, _, _ = compute_sigma_over_mu(y_true, y_pred, bin_edges)
        (S_fit, S_err, C_fit, C_err), res_model = fit_resolution(bc, so_mu, so_mu_err)

        ax1.errorbar(bc, so_mu, yerr=so_mu_err, fmt='o', capsize=4, color=color, label=f"{label}")
        ax1.plot(x_plot, res_model(x_plot, S_fit, C_fit), '--', color=color,
                 label=f"{label} Fit: S={S_fit:.3f}±{S_err:.4f}, C={C_fit:.3f}±{C_err:.4f}")

        fits.append((bc, so_mu, so_mu_err, res_model, S_fit, C_fit))

    # Ratio plot (vs reference)
    bc_ref, so_mu_ref, _, res_model_ref, S_ref, C_ref = fits[ref_idx]
    ref_curve = res_model_ref(x_plot, S_ref, C_ref)

    for i, (bc, so_mu, so_mu_err, res_model, S_fit, C_fit) in enumerate(fits):
        if i == ref_idx: 
            continue
        ratio = res_model(bc, S_fit, C_fit) / res_model_ref(bc, S_ref, C_ref)
        ax2.plot(bc, ratio, 'o-', label=f"Ratio {models_data[i][2]}/{models_data[ref_idx][2]}")

    ax1.set_ylabel("σ / μ")
    ax1.grid(True)
    ax1.legend()
    ax2.axhline(1, color='gray', linestyle='--')
    ax2.set_xlabel("True Energy [GeV]")
    ax2.set_ylabel("Ratio")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("/home/bbapi/model_comparison_new.png")


with open("/home/bbapi/pickles/trueE_target.pickle", "rb") as f:
    targets = pickle.load(f)
targets = np.asarray(targets).reshape(-1, 1)

with open("/home/bbapi/pickles/all_valididx.pickle", "rb") as f:
    valid_ids = pickle.load(f)

# Ensure NumPy arrays
valid_ids = np.array(valid_ids)

y_test = targets[valid_ids]

# Open the file and load its content
with open("/home/bbapi/pickles/A100_test/predictions/pred.pickle", "rb") as file:  # "rb" mode is for reading in binary
    drn_pred = pickle.load(file)

drn_pred = np.array(drn_pred)

bdt_pred = np.load("/home/bbapi/data_splits/best_val_predictions.npy")
dnn_pred = np.load("/home/bbapi/dnn_val_predictions.npy")
drn_pred = drn_pred[valid_ids]

# drn_pred = np.load("")
y_pred_dnn = dnn_pred.ravel()
y_pred_bdt = bdt_pred.ravel()
y_pred_drn = drn_pred.ravel()
y_pred_dnn_np = y_pred_dnn.cpu().numpy() if torch.is_tensor(y_pred_dnn) else y_pred_dnn
y_pred_bdt_np = y_pred_bdt.cpu().numpy() if torch.is_tensor(y_pred_bdt) else y_pred_bdt
y_pred_drn_np = y_pred_drn.cpu().numpy() if torch.is_tensor(y_pred_drn) else y_pred_drn

# y_test is likely still a pandas Series or NumPy array
y_test_np = y_test.values if hasattr(y_test, 'values') else y_test

# # Avoid division by zero
# epsilon = 1e-8
# response_bdt = y_test_np.flatten() / (y_pred_bdt_np.flatten() + epsilon)
# response_dnn = y_test_np.flatten() / (y_pred_dnn_np.flatten() + epsilon)

bin_edges = np.concatenate([
    np.arange(20, 120, 10),
    np.arange(120, targets.max() + 20, 15)
])[:-2]

models_data = [
    (y_test, y_pred_bdt_np, "BDT", "red"),
    (y_test, y_pred_dnn_np, "DNN", "blue"),
    (y_test, y_pred_drn_np, "DRN", "green"),
]

plot_models(models_data, bin_edges, ref_idx=0)





