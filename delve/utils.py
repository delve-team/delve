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


def get_layer_prop(layer, prop, forward_iter=None, save_in_layer=False, **kwargs):
    try: # get prop if already saved in layer
        return getattr(layer, prop)
    except:
        prop = get_prop(layer, prop, **kwargs)
        if save_in_layer:
            layer.prop = prop
        return prop


def get_prop(layer, prop, **kwargs):
    """Low-level function for getting `prop` from `layer`."""
    if prop == 'eig_vals':
        layer_history = get_layer_prop(layer, 'layer_history')
        eig_vals, _ = latent_pca(layer_history, **kwargs)
        return eig_vals


def get_first_representation(batch):
    """Return first instance from minibatch."""
    return batch[0]
