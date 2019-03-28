import time
import torch
from typing import Union, Optional

import numpy as np
from tensorboardX import SummaryWriter

import delve
from delve.metrics import *
from delve.utils import *


def add_eigen_dist(layer, eig_vals=None, n_iter=None):
    if eig_vals is None:
        eig_vals = get_layer_prop(layer, 'eig_vals')
    if n_iter is None:
        n_iter = layer.forward_iter
    layer.writer.add_histogram(
        '{}-eigenvalue_distribution'.format(layer.name),
        eig_vals,
        global_step=n_iter,
        bins=10,
    )
    return eig_vals


def add_neigen_dist(layer, eig_vals=None, n_iter=None):
    if eig_vals is None:
        eig_vals = get_layer_prop(layer, 'eig_vals')
    if n_iter is None:
        n_iter = layer.forward_iter
    eigval_total = sum(eig_vals)
    normalized_eigval_dist = np.array(
        [eigval / eigval_total for eigval in eig_vals])
    layer.writer.add_histogram(
        '{}-normalized_eigenvalue_distribution'.format(layer.name),
        normalized_eigval_dist,
        global_step=n_iter,
        bins=10,
    )
    return eig_vals


def add_saturation_collection(base, layer, saturation_logs):
    base.writer.add_scalars(
        'saturation', saturation_logs, global_step=layer.forward_iter)


def add_layer_saturation(layer, eig_vals=None, n_iter=None, method='cumvar99'):
    if eig_vals is None:
        eig_vals = get_layer_prop(layer, 'eig_vals')
    if n_iter is None:
        n_iter = layer.forward_iter
    nr_eig_vals = get_explained_variance(eig_vals)
    if method == 'cumvar99':
        saturation = get_layer_saturation(nr_eig_vals, layer.out_features)
        layer.writer.add_scalar(f'{layer.name}-percent_saturation-{method}',
                                saturation, n_iter)
    elif method == 'simpson_di':
        saturation = get_eigenval_diversity_index(eig_vals)
        layer.writer.add_scalar(f'{layer.name}-percent_saturation-{method}',
                                saturation, n_iter)
    elif method == 'all':
        cumvar99_saturation = get_layer_saturation(nr_eig_vals,
                                                   layer.out_features)
        layer.writer.add_scalar(f'{layer.name}-percent_saturation-cumvar99',
                                cumvar99_saturation, n_iter)
        simpson_di_saturation = get_eigenval_diversity_index(eig_vals)
        saturation = simpson_di_saturation
        layer.writer.add_scalar(f'{layer.name}-percent_saturation-simpson_di',
                                simpson_di_saturation, n_iter)
    layer.writer.add_scalar(f'{layer.name}-intrinsic_dimensionality',
                            nr_eig_vals, n_iter)
    return eig_vals, saturation


def add_param_eigenvals(layer,
                        eig_vals: Optional[torch.Tensor] = None,
                        top_eigvals: int = 5,
                        n_iter: int = None):
    """Add layer parameter eigenvalues to writer."""
    if eig_vals is None:
        param_eig_vals = get_layer_prop(layer, 'param_eig_vals')
        raise NotImplementedError("Not yet implemented.")
    if n_iter is None:
        n_iter = layer.forward_iter
    param_eig_vals = eig_vals.detach().numpy()
    top_eigvals = min(top_eigvals, len(param_eig_vals))
    layer.writer.add_scalars('{}-parameter_spectrum'.format(layer.name), {
        "param_eig_val{}".format(i): param_eig_vals[i]
        for i in range(top_eigvals)
    }, n_iter)


def add_spectrum(layer,
                 eig_vals: Optional[list] = None,
                 top_eigvals: int = 5,
                 n_iter: int = None):
    """Add layer input eigenvalues to writer."""
    if eig_vals is None:
        eig_vals = get_layer_prop(layer, 'eig_vals')
    if n_iter is None:
        n_iter = layer.forward_iter
    layer.writer.add_scalars(
        '{}-spectrum'.format(layer.name),
        {"eig_val{}".format(i): eig_vals[i]
         for i in range(top_eigvals)},
        n_iter,
    )
    return eig_vals


def add_covariance(layer, activation_batch, n_iter: int):
    layer.writer.add_scalar(
        '{}-latent_representation_covariance'.format(layer.name),
        batch_cov(activation_batch),
        n_iter,
    )


def add_mean(layer, activations_batch, n_iter):
    layer.writer.add_scalar(
        '{}-latent_representation_mean'.format(layer.name),
        batch_mean(activations_batch),
        layer.forward_iter,
    )


def add_spectral_analysis(layer, eig_vals, n_iter, top_eigvals=5):
    """Add spectral analysis `layer` writer and display `top_eigvals`."""
    if eig_vals is None:
        eig_vals = get_layer_prop(layer, 'eig_vals')
    if n_iter is None:
        n_iter = layer.forward_iter
    add_eigen_dist(layer, eig_vals, n_iter)
    add_neigen_dist(layer, eig_vals, n_iter)
    if top_eigvals is not None:
        add_spectrum(layer, eig_vals, top_eigvals, n_iter)
    return eig_vals
