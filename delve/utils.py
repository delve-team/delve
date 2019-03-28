from typing import Any

from .metrics import *


def gen_plot(data, style):
    """Create a pyplot plot and save to buffer."""
    # plt.figure()
    # plt.plot(data)
    # plt.title("test")
    # buf = io.BytesIO()
    # plt.savefig(buf, format='jpeg')
    # buf.seek(0)
    # return buf
    raise NotImplementedError


def get_layer_prop(layer,
                   prop: Any,
                   forward_iter: int = None,
                   save_in_layer: bool = False):
    try:
        return getattr(layer, prop)
    except:
        prop = get_prop(layer, prop)
        if save_in_layer:
            layer.prop = prop
        return prop


def get_prop(layer, prop: Any):
    """Low-level function for getting `prop` from `layer`."""
    if prop == 'eig_vals':
        layer_history = get_layer_prop(layer, 'layer_history')
        # calculate eigenvalues
        eig_vals = latent_pca(layer_history)
        return eig_vals
    elif prop == 'param_eig_vals':
        layer_svd = get_layer_prop(layer, 'layer_svd')
        return layer_svd


def get_first_representation(batch: np.ndarray):
    """Return first instance from minibatch."""
    return batch[0]
