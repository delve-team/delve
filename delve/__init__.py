from .version import __version__

try:
    import torch
    from delve.torchcallback import CheckLayerSat
except ImportError:
    pass
try:
    import keras
    from delve.kerascallback import LayerSaturation, CustomTensorBoard
except ImportError:
    pass

name = "delve"
