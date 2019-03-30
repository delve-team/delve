import torch
from typing import Any

import torch

from .metrics import *


def get_training_state(layer:torch.nn.Module):
    training_state = 'train' if layer.training else 'eval'
    return training_state

def get_layer_prop(layer:torch.nn.Module,
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


def get_prop(layer:torch.nn.Module, prop: Any):
    """Low-level function for getting `prop` from `layer`."""
    training_state = get_training_state(layer)
    if prop in ('train_eig_vals', 'eval_eig_vals'):
        layer_history = get_layer_prop(layer, f'{training_state}_layer_history')
        # calculate eigenvalues
        eig_vals = latent_pca(layer_history)
        return eig_vals
    elif prop == 'param_eig_vals':
        layer_svd = get_layer_prop(layer, 'layer_svd')
        return layer_svd


def get_first_representation(batch: np.ndarray):
    """Return first instance from minibatch."""
    return batch[0]
