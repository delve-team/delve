try:
    import torch
    from delve.torchcallback import CheckLayerSat
except ImportError:
    pass
name = "delve"

from delve.writers import PrintWriter as console
from delve.writers import CSVWriter as csv
from delve.writers import CSVandPlottingWriter as plot
from delve.writers import CSVandPlottingWriter as plotcsv
from delve.writers import CSVandPlottingWriter as csvplot
from delve.writers import TensorBoardWriter as tensorboard