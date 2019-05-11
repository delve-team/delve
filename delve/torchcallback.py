import logging

from collections import OrderedDict
from delve import hooks
from delve.utils import *
from delve.metrics import *
from torch.nn.modules.activation import ReLU
from torch.nn.modules.pooling import MaxPool2d
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.linear import Linear

from tensorboardX import SummaryWriter

logging.basicConfig(format='%(levelname)s:delve:%(message)s',
                    level=logging.INFO)


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
                param_eigvals: eigenvalues of layer parameters

        sat_method         : How to calculate saturation

            Choice of:

                cumvar99   : Proportion of eigenvalues needed to explain 99% of variance
                simpson_di : Simpson diversity index (weighted sum) of explained variance
                             ratios of eigenvalues
                all        : All available methods are logged

        include_conv       : setting to False includes only linear layers
        conv_method        : how to subsample convolutional layers
        verbose (bool)     : print saturation for every layer during training

    """

    def __init__(
            self,
            logging_dir,
            modules,
            log_interval=50,
            min_subsample=128,
            stats: list = ['lsat'],
            include_conv: bool = True,
            conv_method: str = 'median',
            sat_method: str = 'cumvar99',
            verbose=False,
    ):
        self.verbose = verbose
        self.include_conv = include_conv
        self.conv_method = conv_method
        self.sat_method = sat_method
        self.layers = self._get_layers(modules)
        self.min_subsample = min_subsample
        self.writer = self._get_writer(logging_dir)
        self.interval = log_interval
        self.stats = self._check_stats(stats)
        self.logs = {
            'eval-saturation': OrderedDict(),
            'train-saturation': OrderedDict()
        }
        self.global_steps = 0
        self.global_hooks_registered = False
        self.is_notebook = None
        self._init_progress_bar()
        for name, layer in self.layers.items():
            if isinstance(layer, Conv2d) or isinstance(layer, Linear):
                self._register_hooks(layer=layer,
                                     layer_name=name,
                                     interval=log_interval)

    def __exit__(self, exc_type, exc_value, traceback):
        """Called upon closing CheckLayerSat."""
        for bar in self.bars.values():
            bar.close()

    def __getattr__(self, name):
        if name.startswith('add_'):
            return getattr(self.writer, name)
        else:
            try:
                # Redirect to writer object
                return self.writer.__getattribute__(name)
            except:
                # Default behaviour
                return self.__getattribute__(name)

    def __repr__(self):
        return self.layers.keys().__repr__()

    def _init_progress_bar(self):
        try:
            if 'ipykernel.zmqshell.ZMQInteractiveShell' in str(
                    type(get_ipython())):
                from tqdm import tqdm_notebook as tqdm

                self.is_notebook = True
            else:
                from tqdm import tqdm
        except NameError:  # not ipython
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

    def close(self):
        """User endpoint to close writer and progress bars."""
        for bar in self.bars.values():
            bar.close()
        return self.writer.close()

    def _format_saturation(self, saturation_status):
        raise NotImplementedError

    def _write(self, text: str):
        from tqdm import tqdm  # FIXME: Connect to main writer

        tqdm.write("{:^80}".format(text))

    def write(self, text: str):
        self._write(text)

    def saturation(self, is_train=True):
        """User endpoint to get or show saturation levels."""
        return self._show_saturation(is_train)

    def _update(self, layer: torch.nn.Module, percent_sat):
        if self.is_notebook:
            # logging.info("{} - %{} saturated".format(layer, percent_sat))
            self.bars[layer].update(percent_sat)
        else:
            self.bars[layer].update(percent_sat)

    def _show_saturation(self, is_train: bool):
        training_state = 'train' if is_train else 'eval'
        saturation_status = self.logs[f'{training_state}-saturation']
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

    def _check_stats(self, stats: list):
        if not isinstance(stats, list):
            stats = list(stats)
        supported_stats = [
            'lsat',
            'bcov',
            'eigendist',
            'neigendist',
            'spectral',
            'param_eigenvals',
            'spectrum',
        ]
        compatible = [stat in supported_stats for stat in stats]
        incompatible = [i for i, x in enumerate(compatible) if not x]
        assert all(compatible), "Stat {} is not supported".format(
            stats[incompatible[0]])
        return stats

    def _add_conv_layer(self, layer: torch.nn.Module):
        layer.out_features = layer.out_channels
        layer.conv_method = self.conv_method

    def _get_layers(self, modules: Union[list, torch.nn.Module]):
        layers = {}
        if not isinstance(modules, list) and not hasattr(
                modules, 'out_features'):
            # is a model with layers
            # check if submodule
            submodules = modules._modules  # OrderedDict
            for idx, (name, submodule) in enumerate(submodules.items()):
                if submodule._get_name() is 'Sequential':
                    for submodule_idx, layer in submodule._modules.items():
                        layer_type = layer._get_name().lower()
                        if layer_type == 'conv2d':
                            if self.include_conv:
                                self._add_conv_layer(layer)
                            else:
                                continue
                        layers[name + submodule_idx] = layer
                else:
                    layer_name = name.split('.')[0]
                    layer_type = submodule._get_name().lower()
                    if not layer_type in ['conv2d', 'linear']:
                        print(f"Skipping {layer_type}")
                        continue
                    layer = getattr(modules, layer_name)
                    if layer_type == 'conv2d':
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
                if layer_type == 'conv2d':
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
            return layers

    def _get_writer(self, writer_dir):
        """Create a writer to log history to `writer_dir`."""
        writer = SummaryWriter(writer_dir)
        writer_name = list(writer.all_writers.keys())[0]  # eg, linear1
        if self.verbose:
            logging.info(
                "Adding summaries to directory: {}".format(writer_name))
        return writer

    def _register_hooks(self, layer: torch.nn.Module, layer_name: str,
                        interval):
        if not hasattr(layer, 'eval_layer_history'):
            layer.eval_layer_history = []
        if not hasattr(layer, 'train_layer_history'):
            layer.train_layer_history = []
        if not hasattr(layer, 'layer_svd'):
            layer.layer_svd = None
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

    def register_backward_hooks(self, layer: torch.nn.Module, stats):
        """Register hook to show `stats` in `layer`."""

        # HACK: Update global changes via arbitrary layer on backwards pass
        def global_saturation_update(layer, input, output):
            """Hook to register in `layer` module."""
            if layer.forward_iter % layer.interval == 0:
                training_state = get_training_state(layer)
                hooks.add_saturation_collection(
                    self, layer, self.logs[f'{training_state}-saturation'])

        if not self.global_hooks_registered:
            if 'lsat' in stats:
                layer.register_backward_hook(global_saturation_update)
            self.global_hooks_registered = True

    def register_forward_hooks(self, layer: torch.nn.Module, stats: list):
        """Register hook to show `stats` in `layer`."""

        def record_layer_saturation(layer: torch.nn.Module, input, output):
            """Hook to register in `layer` module."""

            # Increment step counter
            layer.forward_iter += 1
            if layer.forward_iter % layer.interval == 0:
                activations_batch = output.data.cpu().numpy()
                training_state = 'train' if layer.training else 'eval'
                layer_history = setattr(layer,
                                        f'{training_state}_layer_history',
                                        activations_batch)
                #layer_history.append(activations_batch)
                eig_vals = None
                if 'bcov' in stats:
                    hooks.add_covariance(layer, activations_batch,
                                         layer.forward_iter)
                if 'mean' in stats:
                    hooks.add_mean(layer, activations_batch,
                                   layer.forward_iter)
                if 'spectral' in stats:
                    if (layer.forward_iter %
                        (layer.interval *
                         10) == 0):  # expensive spectral analysis
                        eig_vals = hooks.add_spectral_analysis(
                            layer, eig_vals, layer.forward_iter)
                if 'lsat' in stats:
                    eig_vals, saturation = hooks.add_layer_saturation(
                        layer, eig_vals=eig_vals, method=self.sat_method)
                    training_state = 'train' if layer.training else 'eval'

                    self.logs[f'{training_state}-saturation'][
                        layer.name] = saturation
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
                if 'param_eigenvals' in stats:
                    layer_param_weight = layer.state_dict()['weight']
                    u, s, v = layer_param_weight.svd()
                    param_eigenvals = hooks.add_param_eigenvals(
                        layer, s, top_eigvals=5, n_iter=layer.forward_iter)

        layer.register_forward_hook(record_layer_saturation)
