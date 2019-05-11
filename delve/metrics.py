from typing import Union

import numpy as np
from mdp.utils import CovarianceMatrix
from typing import Dict
from mdp.utils import CovarianceMatrix
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


def _get_cov(layer_history: Union[list, np.ndarray],
             subsample_rate: int = 50,
             window_size: int = 100,
             conv_method: str = 'median'):
    """Get covariance matrix of layer activation history.
    Args:
        subsample_rate : int, subsample rate before calculating PCs
        window_size    : int, how many activations to use for calculating principal components
        conv_method    : str, method for sampling convolutional layer, eg, median
    """
    history_array = np.vstack(layer_history[-window_size:])  # list to array

    if len(history_array.shape) == 4:  # conv layer (B x C x H x W)
        if conv_method == 'median':
            history_array = np.median(history_array,
                                      axis=(2, 3))  # channel median
        elif conv_method == 'max':
            history_array = np.max(history_array,
                                   axis=(2, 3))  # channel median
        elif conv_method == 'mean':
            history_array = np.mean(history_array,
                                    axis=(2, 3))  # channel median
    history_array = history_array.reshape(history_array.shape[0], -1)
    assert (len(history_array.shape) is
            2), "Stacked layer history shape is {}, \
        should be 2".format(history_array.shape)
    embeddings = np.vstack(
        history_array)[::subsample_rate]  # subsample every Nth representation
    cov = np.cov(embeddings.T)
    return cov


def _get_iterative_cov(layer, batch, conv_method: str = 'median'):

    #batch = batch[-1]

    if len(batch.shape) == 4:  # conv layer (B x C x H x W)
        if conv_method == 'median':
            batch = np.median(batch, axis=(2, 3))  # channel median
        elif conv_method == 'max':
            batch = np.max(batch, axis=(2, 3))  # channel median
        elif conv_method == 'mean':
            batch = np.mean(batch, axis=(2, 3))

    if not layer in COVARIANCE_MATRICES:
        COVARIANCE_MATRICES[layer] = CovarianceMatrix()
        COVARIANCE_MATRICES[layer]._init_internals(batch)
    else:
        COVARIANCE_MATRICES[layer].update(batch)
    return COVARIANCE_MATRICES[layer]._cov_mtx


def latent_pca(layer_history: list, conv_method: str = 'median'):
    """Get NxN matrix of principal components sorted in descending order from `layer_history`
    Args:
        layer_history : list, layer outputs during training
        conv_method   : method for sampling convolutional layer input if layer is conv2D
    Returns:
        eig_vals       : numpy.ndarray of absolute value of eigenvalues, sorted in descending order
        P              : numpy.ndarray, NxN square matrix of principal components calculated over training

    """
    cov = _get_cov(layer_history, conv_method=conv_method)
    eig_vals = np.linalg.eigvalsh(cov)

    # Sort the eigenvalues from high to low
    eig_vals = sorted(eig_vals, reverse=True)
    return eig_vals


def latent_iterative_pca(layer, batch, conv_method: str = 'median'):
    """Get NxN matrix of principal components sorted in descending order from `layer_history`
    Args:
        layer_history : list, layer outputs during training
    Returns:
        eig_vals       : numpy.ndarray of absolute value of eigenvalues, sorted in descending order
        P              : numpy.ndarray, NxN square matrix of principal components calculated over training

    """
    cov = _get_iterative_cov(layer, batch, conv_method=conv_method)
    eig_vals = np.linalg.eigvalsh(cov)

    # Sort the eigenvalues from high to low
    eig_vals = sorted(eig_vals, reverse=True)
    return eig_vals


def latent_svd(layer_history: list):
    """Get NxN matrix of sorted (largest to smallest) singular values from `layer_history`
    Args:
        layer_history : list, layer outputs during training
    Returns:
        eig_vals  : numpy.ndarray of eigenvalues, sorted in descending order
        eig_vecs   : numpy.ndarray, NxN square matrix of eigenvectors calculated over training

    NOTE: Method is experimental.
    """
    cov = _get_cov(layer_history)
    u, s, eig_vecs = np.linalg.svd(cov)
    eig_vals = s**2
    return eig_vals, eig_vecs


def batch_mean(batch: np.ndarray):
    """Get mean of first vector in `batch`."""  # TODO: Add support for non-dense layers.
    return np.mean(batch[0])


def batch_cov(batch: np.ndarray):
    """Get covariance of first instance in `batch`."""  # TODO: Add support for non-dense layers.
    return np.cov(batch[0])
