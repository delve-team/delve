import delve
import numpy as np
import time

from delve.metrics import *
from delve.utils import *
from tensorboardX import SummaryWriter


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
    normalized_eigval_dist = np.array([eigval / eigval_total for eigval in eig_vals])
    layer.writer.add_histogram(
        '{}-normalized_eigenvalue_distribution'.format(layer.name),
        normalized_eigval_dist,
        global_step=n_iter,
        bins=10,
    )
    return eig_vals


def add_saturation_collection(base, layer, saturation_logs):
    base.writer.add_scalars(
        'saturation', saturation_logs, global_step=layer.forward_iter
    )


def add_layer_saturation(layer, eig_vals=None, n_iter=None):
    if eig_vals is None:
        eig_vals = get_layer_prop(layer, 'eig_vals')
    if n_iter is None:
        n_iter = layer.forward_iter
    nr_eig_vals = get_explained_variance(eig_vals)
    saturation = get_layer_saturation(nr_eig_vals, layer.out_features)
    layer.writer.add_scalar(
        '{}-intrinsic_dimensionality'.format(layer.name), nr_eig_vals, n_iter
    )
    layer.writer.add_scalar(
        '{}-percent_saturation'.format(layer.name), saturation, n_iter
    )
    return eig_vals, saturation


def add_spectrum(layer, eig_vals=None, top_eigvals=5, n_iter=None):
    if eig_vals is None:
        eig_vals = get_layer_prop(layer, 'eig_vals')
    if n_iter is None:
        n_iter = layer.forward_iter
    layer.writer.add_scalars(
        '{}-spectrum'.format(layer.name),
        {"eig_val{}".format(i): eig_vals[i] for i in range(top_eigvals)},
        n_iter,
    )
    return eig_vals


def add_covariance(layer, activation_batch, n_iter):
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
