import logging
import numpy as np
import time

import delve
from delve import hooks
from delve.utils import *
from delve.metrics import *

from tensorboardX import SummaryWriter

logging.basicConfig(format='%(levelname)s:delve:%(message)s', level=logging.INFO)

class CheckLayerSat(object):
    """Takes PyTorch layers, layer or model as `modules` and writes tensorboardX
    summaries to `logging_dir`.
    """

    def __init__(self, logging_dir, modules, log_interval=10, stats=['lsat']):
        self.layers = self._get_layers(modules)
        self.writer = self._get_writer(logging_dir)
        self.interval = log_interval
        self.stats = stats

        for name, layer in self.layers.items():
            self._register_hooks(
                layer=layer, layer_name=name, interval=log_interval)

    def __getattr__(self, name):
        if name.startswith('add_'):
            return getattr(self.writer, name)
        elif name == 'close':
            return getattr(self.writer, name)
        else:
            # Default behaviour
            return self.__getattribute__(name)

    def __repr__(self):
        return self.layers.keys().__repr__()

    def __check_stats(self, stats):
        supported_stats = [
            'lsat', 'bcov', 'eigendist', 'neigendist', 'spectral', 'spectrum'
        ]
        compatible = [stat in supported_stats for stat in stats]
        incompatible = [i for i, x in compatible if not x][0]
        assert all(compatible), "Stat {} is not supported".format(
            stats[incompatible])

    def _get_layers(self, modules):
        layers = {}
        if not isinstance(modules, list) and not hasattr(
                modules, 'out_features'):
            # is a model with layers
            for k, v in modules.state_dict().items():
                layer_name = k.split('.')[0]
                layers[layer_name] = module.__getattr__(layer_name)
            return layers
        elif isinstance(modules, list):  # FIXME: Optimize dictionary creation
            layer_names = []
            for layer in modules:
                layer_class = layer.__module__.split('modules')[-1]
                layer_names.append(layer_class)
                layer_cnt = layer_names.count(layer_class)
                layer_name = layer_class + str(layer_cnt)
                layers[layer_name] = layer
            logging.info("Recording layers {}".format(layers))
            return layers

    def _get_writer(self, writer_dir):
        """Create a writer to log history to `writer_dir`."""
        writer = SummaryWriter(writer_dir)
        writer_name = list(writer.all_writers.keys())[0]  # eg, linear1
        logging.info("Adding summaries to directory: {}".format(writer_name))
        return writer

    def _register_hooks(self, layer, layer_name, interval):
        if not hasattr(layer, 'latent_history'):
            layer.latent_history = []
        if not hasattr(layer, 'forward_iter'):
            layer.forward_iter = 0
        if not hasattr(layer, 'interval'):
            layer.interval = interval
        if not hasattr(layer, 'writer'):
            layer.writer = self.writer
        if not hasattr(layer, 'name'):
            layer.name = layer_name
        self.register_forward_hooks(layer, self.stats)
        return self

    def register_forward_hooks(self, layer, stats):
        """Register hook to show `stats` in `layer`."""

        def record_layer_saturation(layer, input, output):
            """Hook to register in `layer` module."""

            # Increment step counter
            layer.forward_iter += 1
            if layer.forward_iter % layer.interval == 0:
                activations_batch = output.data.cpu().numpy()
                layer.latent_history.append(activations_batch)
                activations_vec = get_first_representation(activations_batch)
                eig_vals = None
                if 'bcov' in stats:
                    hooks.add_covariance(layer, activations_batch,
                                         layer.forward_iter)
                if 'mean' in stats:
                    hooks.add_mean(layer, activations_batch,
                                   layer.forward_iter)
                if 'spectral' in stats:
                    if layer.forward_iter % (
                            layer.interval *
                            10) == 0:  # expensive spectral analysis
                        eig_vals = hooks.record_spectral_analysis(
                            layer, eig_vals, layer.forward_iter)
                if 'lsat' in stats:
                    eig_vals = hooks.add_layer_saturation(
                        layer, eig_vals=eig_vals)
                if 'spectral' not in stats:
                    if 'eigendist' in stats:
                        eig_vals = hooks.add_eigen_dist(
                            layer, eig_vals, layer.forward_iter)
                    if 'neigendist' in stats:
                        eig_vals = hooks.add_neigen_dist(
                            layer, eig_vals, layer.forward_iter)
                    if 'spectrum' in stats:
                        eig_vals = hooks.add_spectrum(
                            layer,
                            eig_vals,
                            top_eigvals=5,
                            n_iter=layer.forward_iter)

        layer.register_forward_hook(record_layer_saturation)
