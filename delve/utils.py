from typing import Any

import numpy as np
import torch
from delve.metrics import latent_pca, latent_iterative_pca


def get_training_state(layer: torch.nn.Module):
    training_state = 'train' if layer.training else 'eval'
    return training_state


def get_layer_prop(layer: torch.nn.Module,
                   prop: Any,
                   forward_iter: int = None,
                   save_in_layer: bool = False):
    try:
        return getattr(layer, prop)
    except AttributeError:
        prop = get_prop(layer, prop)
        if save_in_layer:
            layer.prop = prop
        return prop


def get_prop(layer: torch.nn.Module, prop: Any):
    """Low-level function for getting `prop` from `layer`."""
    training_state = get_training_state(layer)
    if prop in ('train_eig_vals', 'eval_eig_vals'):
        layer_history = get_layer_prop(layer,
                                       f'{training_state}_layer_history')
        # calculate eigenvalues

        if hasattr(layer, 'conv_method'):
            eig_vals = latent_iterative_pca(layer,
                                            layer_history,
                                            conv_method=layer.conv_method)
        else:
            eig_vals = latent_iterative_pca(layer, layer_history)
        return eig_vals
    elif prop == 'param_eig_vals':
        layer_svd = get_layer_prop(layer, 'layer_svd')
        return layer_svd


def get_first_representation(batch: np.ndarray):
    """Return first instance from minibatch."""
    return batch[0]
