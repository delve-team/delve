from typing import Optional

import numpy as np
import torch
from delve.metrics import get_explained_variance, get_layer_saturation, get_eigenval_diversity_index, batch_cov, \
    batch_mean, _get_iterative_cov
from mdp.utils import CovarianceMatrix

from delve.utils import get_layer_prop, get_training_state

def add_layer_saturation(layer: torch.nn.Module,
                         eig_vals: Optional[np.ndarray] = None,
                         n_iter: Optional[int] = None,
                         method='cumvar99'):
    training_state = get_training_state(layer)
    layer_type = layer._get_name().lower()

    if eig_vals is None:
        eig_vals = get_layer_prop(layer, f'{training_state}_eig_vals')
    if n_iter is None:
        n_iter = layer.forward_iter
    nr_eig_vals = get_explained_variance(eig_vals)

    layer_name = layer.name + (f'_{layer.conv_method}'
                               if layer_type == 'conv2d' else '')
    if method == 'cumvar99':
        saturation = get_layer_saturation(nr_eig_vals, layer.out_features)
        layer.writer.add_scalar(
            f'{training_state}-{layer_name}-percent_saturation-{method}',
            saturation, n_iter)
    elif method == 'simpson_di':
        saturation = get_eigenval_diversity_index(eig_vals)
        layer.writer.add_scalar(
            f'{training_state}-{layer_name}-percent_saturation-{method}',
            saturation, n_iter)
    elif method == 'all':
        cumvar99_saturation = get_layer_saturation(nr_eig_vals,
                                                   layer.out_features)
        layer.writer.add_scalar(
            f'{training_state}-{layer_name}-percent_saturation-cumvar99',
            cumvar99_saturation, n_iter)
        simpson_di_saturation = get_eigenval_diversity_index(eig_vals)
        saturation = simpson_di_saturation
        layer.writer.add_scalar(
            f'{training_state}-{layer_name}-percent_saturation-simpson_di',
            simpson_di_saturation, n_iter)
    layer.writer.add_scalar(
        f'{training_state}-{layer_name}-intrinsic_dimensionality', nr_eig_vals,
        n_iter)
    return eig_vals, saturation
