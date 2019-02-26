import sys
import delve
import logging
import numpy as np
import time

from collections import OrderedDict
from delve import hooks
from delve.utils import *
from delve.metrics import *
from tensorboardX import SummaryWriter

logging.basicConfig(format='%(levelname)s:delve:%(message)s', level=logging.INFO)


class CheckLayerSat(object):
    """Takes PyTorch layers, layer or model as `modules` and writes tensorboardX
    summaries to `logging_dir`. Outputs layer saturation with call to self.saturation().

    Args:
        logging_dir (str)  : destination for summaries
        modules (torch modules or list of modules) : layer-containing object
        log_interval (int) : steps between writing summaries
        stats (list of str): list of stats to collect

            supported stats are:
                lsat       : layer saturation
                bcov       : batch covariance
                eigendist  : eigenvalue distribution
                neigendist : normalized eigenvalue distribution
                spectrum   : top-N eigenvalues of covariance matrix
                spectral   : spectral analysis (eigendist, neigendist, and spectrum)

        include_conv       : bool, setting to False includes only linear layers
        verbose (bool)     : print saturation for every layer during training

    """

    def __init__(
        self,
        logging_dir,
        modules,
        log_interval=50,
        stats=['lsat'],
        include_conv=True,
        verbose=False,
    ):
        self.verbose = verbose
        self.include_conv = include_conv
        self.layers = self._get_layers(modules)
        self.writer = self._get_writer(logging_dir)
        self.interval = log_interval
        self.stats = self._check_stats(stats)
        self.logs = {'saturation': OrderedDict()}
        self.global_steps = 0
        self.global_hooks_registered = False
        self.is_notebook = None
        self._init_progress_bar()
        for name, layer in self.layers.items():
            self._register_hooks(layer=layer, layer_name=name, interval=log_interval)

    def __exit__(self, exc_type, exc_value, traceback):
        """Called upon closing CheckLayerSat."""
        for bar in self.bars.values():
            bar.close()

    def __getattr__(self, name):
        if name.startswith('add_'):
            return getattr(self.writer, name)
        else:
            # Default behaviour
            return self.__getattribute__(name)

    def __repr__(self):
        return self.layers.keys().__repr__()

    def _init_progress_bar(self):
        try:
            if 'ipykernel.zmqshell.ZMQInteractiveShell' in str(type(get_ipython())):
                from tqdm import tqdm_notebook as tqdm

                self.is_notebook = True
        except:
            from tqdm import tqdm

            self.is_notebook = False
        bars = {}
        for i, layer in enumerate(self.layers.keys()):
            # bar_format = "{l_bar}{bar}| {n:.3g}/{total_fmt} [{rate_fmt}{postfix}]" # FIXME: Make it prettier
            pbar = tqdm(desc=layer, total=100, leave=True, position=i + 1)
            bars[layer] = pbar
            # bar = ChargingBar('{} Saturation'.format(layer), suffix='%(percent)d%%')
            # bars[layer] = bar
        self.bars = bars

    def get_data(self):
        raise NotImplementedError

    def close(self):
        """User endpoint to close writer and progress bars."""
        for bar in self.bars.values():
            bar.close()
        return self.writer.close()

    def _format_saturation(self, saturation_status):
        raise NotImplementedError

    def _write(self, text):
        from tqdm import tqdm  # FIXME: Connect to main writer

        tqdm.write("{:^80}".format(text))

    def write(self, text):
        self._write(text)

    def saturation(self):
        """User endpoint to get or show saturation levels."""
        return self._show_saturation()

    def _update(self, layer, percent_sat):
        if self.is_notebook:
            # logging.info("{} - %{} saturated".format(layer, percent_sat))
            self.bars[layer].update(percent_sat)
        else:
            self.bars[layer].update(percent_sat)

    def _show_saturation(self):
        saturation_status = self.logs['saturation']
        # saturation_status = 69
        for layer, saturation in saturation_status.items():
            curr = self.bars[layer].n
            percent_sat = int(max(0, saturation - curr))
            self._update(layer, percent_sat)
        # # Global saturations # NOTIMPLEMENTED
        # saturations = [s for s in saturation_status.values()]
        # if len(saturations):
        #     for bar in self.bars:
        #         stack.index = sum(saturations)/len(saturations)
        #         stack.update()
        return saturation_status

    def _check_stats(self, stats):
        if not isinstance(stats, list):
            stats = list(stats)
        supported_stats = [
            'lsat',
            'bcov',
            'eigendist',
            'neigendist',
            'spectral',
            'spectrum',
        ]
        compatible = [stat in supported_stats for stat in stats]
        incompatible = [i for i, x in enumerate(compatible) if not x]
        assert all(compatible), "Stat {} is not supported".format(
            stats[incompatible[0]]
        )
        return stats

    def _add_conv_layer(self, layer):
        layer.out_features = layer.out_channels

    def _get_layers(self, modules):
        layers = {}
        if not isinstance(modules, list) and not hasattr(modules, 'out_features'):
            # is a model with layers
            # check if submodule
            submodules = modules._modules # OrderedDict
            for idx, (name, submodule) in enumerate(submodules.items()):
                if submodule._get_name() is 'Sequential':
                    for submodule_idx, layer in submodule._modules.items():
                        layer_class = layer.__module__.split('.')[-1]
                        if layer_class == 'conv':
                            if self.include_conv:
                                self._add_conv_layer(layer)
                            else:
                                continue
                        layers[name+submodule_idx] = layer
                else:
                    layer_name = name.split('.')[0]
                    layer_type = submodule._get_name().lower()
                    if not layer_type in ['conv','linear']:
                        continue
                    layer = getattr(modules, layer_name)
                    layer_class = layer.__module__.split('.')[-1]
                    if layer_class == 'conv':
                        if self.include_conv:
                            self._add_conv_layer(layer)
                        else:
                            continue
                    layers[layer_name] = layer
            return layers
        elif isinstance(modules, list):
            # is a list of layers
            layer_names = []
            for layer in modules:
                try:
                    layer_class = layer.__module__.split('.')[-1]
                except:
                    raise "Layer {} is not supported".format(layer)
                if layer_class == 'conv':
                    if self.include_conv:
                        self._add_conv_layer(layer)
                    else:
                        continue
                layer_names.append(layer_class)
                layer_cnt = layer_names.count(layer_class)
                layer_name = layer_class + str(layer_cnt)
                layers[layer_name] = layer
            if self.verbose:
                logging.info("Recording layers {}".format(layers))
        # elif layer._get_name() is 'Sequential':
        #                 for idx, l in enumerate(layer):
        #                     self._register_hooks(layer=l, layer_name=name+str(idx), interval=log_interval)
            return layers

    def _get_writer(self, writer_dir):
        """Create a writer to log history to `writer_dir`."""
        writer = SummaryWriter(writer_dir)
        writer_name = list(writer.all_writers.keys())[0]  # eg, linear1
        if self.verbose:
            logging.info("Adding summaries to directory: {}".format(writer_name))
        return writer

    def _register_hooks(self, layer, layer_name, interval):
        if not hasattr(layer, 'layer_history'):
            layer.layer_history = []
        if not hasattr(layer, 'forward_iter'):
            layer.forward_iter = 0
        if not hasattr(layer, 'interval'):
            layer.interval = interval
        if not hasattr(layer, 'writer'):
            layer.writer = self.writer
        if not hasattr(layer, 'name'):
            layer.name = layer_name
        self.register_forward_hooks(layer, self.stats)
        self.register_backward_hooks(layer, self.stats)
        return self

    def register_backward_hooks(self, layer, stats):
        """Register hook to show `stats` in `layer`."""
        # HACK: Update global changes via arbitrary layer on backwards pass
        def global_saturation_update(layer, input, output):
            """Hook to register in `layer` module."""
            if layer.forward_iter % layer.interval == 0:
                hooks.add_saturation_collection(self, layer, self.logs['saturation'])

        if not self.global_hooks_registered:
            if 'lsat' in stats:
                layer.register_backward_hook(global_saturation_update)
            self.global_hooks_registered = True

    def register_forward_hooks(self, layer, stats):
        """Register hook to show `stats` in `layer`."""

        def record_layer_saturation(layer, input, output):
            """Hook to register in `layer` module."""

            # Increment step counter
            layer.forward_iter += 1
            if layer.forward_iter % layer.interval == 0:
                activations_batch = output.data.cpu().numpy()
                layer.layer_history.append(activations_batch)
                activations_vec = get_first_representation(activations_batch)
                eig_vals = None
                if 'bcov' in stats:
                    hooks.add_covariance(layer, activations_batch, layer.forward_iter)
                if 'mean' in stats:
                    hooks.add_mean(layer, activations_batch, layer.forward_iter)
                if 'spectral' in stats:
                    if (
                        layer.forward_iter % (layer.interval * 10) == 0
                    ):  # expensive spectral analysis
                        eig_vals = hooks.add_spectral_analysis(
                            layer, eig_vals, layer.forward_iter
                        )
                if 'lsat' in stats:
                    eig_vals, saturation = hooks.add_layer_saturation(
                        layer, eig_vals=eig_vals
                    )
                    self.logs['saturation'][layer.name] = saturation
                if 'spectral' not in stats:
                    if 'eigendist' in stats:
                        eig_vals = hooks.add_eigen_dist(
                            layer, eig_vals, layer.forward_iter
                        )
                    if 'neigendist' in stats:
                        eig_vals = hooks.add_neigen_dist(
                            layer, eig_vals, layer.forward_iter
                        )
                    if 'spectrum' in stats:
                        eig_vals = hooks.add_spectrum(
                            layer, eig_vals, top_eigvals=5, n_iter=layer.forward_iter
                        )

        layer.register_forward_hook(record_layer_saturation)


