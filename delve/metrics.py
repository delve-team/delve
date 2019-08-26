from typing import Dict, Union

import numpy as np

COVARIANCE_MATRICES = dict()


def get_eigenval_diversity_index(eig_vals: np.ndarray):
    """Return Simpson diversity index of eigvalue explained variance ratios.
    Args:
        eig_vals   : eigenvalues in descending order

    """
    tot = sum(eig_vals)
    var_exp = [(i / tot) for i in eig_vals]
    simpson_di = sum(x**2 for x in var_exp) * 100
    return simpson_di


def get_explained_variance(eig_vals: np.ndarray,
                           threshold: float = 0.99,
                           return_cum: bool = False):
    """Return number of `eig_vals` needed to explain `threshold`*100 percent of variance
    Args:
        eig_vals   : numpy.ndarray of eigenvalues in descending order
        threshold  : percent of variance to explain
        return_cum : return the cumulative array of eigenvalues
    Return:
        nr_eig_vals : int
        cum_var_exp : numpy.ndarray

            or

        nr_eig_vals : int
    """
    tot = sum(eig_vals)
    var_exp = [(i / tot) for i in eig_vals]
    cum_var_exp = np.cumsum(var_exp)
    nr_eig_vals = np.argmax(cum_var_exp > threshold)
    if return_cum:
        return nr_eig_vals, cum_var_exp
    return nr_eig_vals


def get_layer_saturation(nr_eig_vals: np.ndarray, layer_width: int) -> int:
    saturation = round(100 * nr_eig_vals / layer_width, 2)
    return saturation

def batch_mean(batch: np.ndarray):
    """Get mean of first vector in `batch`."""  # TODO: Add support for non-dense layers.
    return np.mean(batch[0])

def batch_cov(batch: np.ndarray):
    """Get covariance of first instance in `batch`."""  # TODO: Add support for non-dense layers.
    return np.cov(batch[0])

def compute_saturation(cov: np.ndarray, thresh: float = .99) -> float:
    """
    Provides layer saturation
    :param covariance_matrix:
    :param thresh:
    :param conv_method:
    :return:
    """
    
    eig_vals = np.linalg.eigvalsh(cov)

    # Sort the eigenvalues from high to low
    eig_vals = sorted(eig_vals, reverse=True)
    total_dim = len(cov)
    nr_eigs = get_explained_variance(eig_vals=eig_vals, threshold=thresh, return_cum=False)
    return get_layer_saturation(nr_eig_vals=nr_eigs, layer_width=total_dim)
