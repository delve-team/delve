import logging
from typing import List, Dict, Any, Optional, Union
import torch
from collections import OrderedDict
from itertools import product
from torch.nn.modules.activation import ReLU
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.linear import Linear
from torch.nn.modules import LSTM
#from mdp.utils import CovarianceMatrix
from delve.torch_utils import TorchCovarianceMatrix
from delve.writers import CompositWriter, NPYWriter, STATMAP
from delve.metrics import *
import delve
import warnings

logging.basicConfig(format='%(levelname)s:delve:%(message)s',
                    level=logging.INFO)


class CheckLayerSat(object):
    """Takes PyTorch module and records layer saturation, intrinsic dimensionality and other scalars.

    Args:
        savefile (str)  : destination for summaries
        save_to (str, List[Union[str, delve.writers.AbstractWriter]]: Specify one or multiple save strategies.
        You can use preimplemented save strategies or inherit from the AbstractWriter in order to implement your
        own preferred saving strategy.

            preixisting saving strategies are:
                csv         : stores all stats in a csv-file with 1 row for each epoch.
                plot        : produces plots from intrinsic dimensionality and / or layer saturation
                tensorboard : saves all stats to tensorboard
                print       : print all metrics on console as soon as they are logged
                npy         : creates a folder-structure with npy-files containing the logged values. This is the only
                              save strategy that can save the full covariance matrix.
                              This strategy is useful if you want to reproduce intrinsic dimensionality and saturation
                              values with other thresholds without re-evaluating model checkpoints.
        modules (torch modules or list of modules) : layer-containing object. Per default, only Conv2D, Linear and LSTM-Cells are recorded
        writers_args (dict) : contains additional arguments passed over to the writers. This is only used, when a writer is initialized
                              through a string-key.
        log_interval (int) : distances between two batches used for updaing the covariance matrix. Default value is 1, which means
                             that all data is used for computing intrinsic dimensionality and saturation. Increasing the log
                             interval is usefull on very large datasets to reduce numeric instability.
        max_samples (int)  : if this value is set, the covariance matrix in each layer will halt updating itself when max_samples
                             are reached. Usecase is similar to log-interval, when datasets are very large.
        stats (list of str): list of stats to compute

            supported stats are:
                idim        : intrinsic dimensionality
                lsat        : layer saturation (intrinsic dimensionality divided by feature space dimensionality)
                cov         : the covariance-matrix (only saveable using the 'npy' save strategy)

        layerwise_sat (bool): weather or not to include layerwise saturation when saving
        reset_covariance (bool): True by default, resets the covariance everytime the stats are computed. Disabling this,
                            will cause a strong bias of the covariance due to gradient influencing the model. We recomment
                            computing saturation at the end of training and testing.

        include_conv       : setting to False includes only linear layers
        conv_method (str)  : how to subsample convolutional layers. Default is channelwise, which means that the
                             each position of the filter tensor is considered a datapoint, effectivly yielding
                             a data matrix of shape (height*width*batch_size, num_filters)

            supported stats are:
                channelwise : treats every depth vector of the tensor as a datapoint, effectivly reshaping the
                              data tensor from shape (batch_size, height, width, channel) into (batch_size*height*width, channel).
                mean        : applies global average pooling on each feature map
                max         : applies global max pooling on each feature map
                median      : applies global median pooling on each feature map
                flatten     : flattenes the entire feature map to a vector, reshaping the data tensor into a data matrix
                              of shape (batch_size, height*width*channel). This strategy for dealing with convolutions is extremly
                              memory intensive and will likely cause memory and performance problems for any non toy-problem
        verbose (bool)     : print saturation for every layer during training
        sat_threshold (float): the threshold used to determine the number of eigendirection belonging to the latent space
                                in effects, this is the threshold determining the the intrinsic dimensionality. Default value
                                is 0.99 (99% of the explained variance), which is the best compromise between a good and interpretable approximation.
                                From experience the threshold should be between 0.97 and 0.9995 for meaningfull results.
        verbose (bool):     Change verbosity level (default is 0)
        device (str)       :   Device to do the computations on. Default is cuda:0. Generally it is recommended to do the computations
                            on gpu in order to get maximum performance. Using the cpu is generally slower but it enables
                            delve to use regular RAM instead of the generally more limited VRAM of the GPU.
                            Not having delve run on the same device as the network causes slight performance decrease due
                            to copying memory between devices during each forward pass.
                            Delve can handle models distributed on multiple GPUs, however delve itself will always run on a single device.

    """

    def __init__(
            self,
            savefile: str,
            save_to: Union[str, delve.writers.AbstractWriter],
            modules: torch.nn.Module,
            writer_args: Optional[Dict[str, Any]] = None,
            log_interval=1,
            max_samples=None,
            stats: list = ['lsat'],
            layerwise_sat: bool = True,
            reset_covariance: bool = True,
            average_sat: bool = False,
            ignore_layer_names: List[str] = [],
            include_conv: bool = True,
            conv_method: str = 'channelwise',
            sat_threshold: str = .99,
            verbose=False,
            device='cuda:0',
    ):
        self.verbose = verbose
        self.include_conv = include_conv
        self.conv_method = conv_method
        self.threshold = sat_threshold
        self.layers = self.get_layers_recursive(modules)
        self.max_samples = max_samples
        self.log_interval = log_interval
        self.reset_covariance = reset_covariance

        if writer_args is None:
            writer_args = {}
        writer_args['savepath'] = savefile

        self.writer = self._get_writer(save_to, writer_args)
        self.interval = log_interval

        self._warn_if_covariance_not_saveable(stats)

        self.logs, self.stats = self._check_stats(stats)
        self.layerwise_sat = layerwise_sat
        self.average_sat = average_sat
        self.ignore_layer_names = ignore_layer_names
        self.seen_samples = {
            'train': {},
            'eval': {}
        }
        self.global_steps = 0
        self.global_hooks_registered = False
        self.is_notebook = None
        self.device = device
        for name, layer in self.layers.items():
            if isinstance(layer, Conv2d) or isinstance(layer, Linear) or isinstance(layer, LSTM):
                self._register_hooks(layer=layer,
                                     layer_name=name,
                                     interval=log_interval)

    def _warn_if_covariance_not_saveable(self, stats: List[str]):
        warn = False
        if 'cov' in stats:
            if isinstance(self.writer, CompositWriter):
                for writer in self.writer.writers:
                    if isinstance(writer, NPYWriter):
                        return
                warn = True
            elif not isinstance(self.writer, NPYWriter):
                warn = True
        if warn:
            warnings.warn("'cov' was selected as stat, but 'npy' (NPYWriter) is not used as a save strategy,"
                          "which is the only writer able to save the covariance matrix. The training and logging will"
                          "run normally, but the covariance matrix will not be saved. Note that you can add multiple writers"
                          "by passing a list.")


    def __getattr__(self, name):
        if name.startswith('add_') and name != 'add_saturations':
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

    def close(self):
        """User endpoint to close writer and progress bars."""
        return self.writer.close()

    def _format_saturation(self, saturation_status):
        raise NotImplementedError

    def _check_stats(self, stats: list):
        if not isinstance(stats, list):
            stats = list(stats)
        supported_stats = [
            'lsat',
            'idim',
            'cov',
        ]
        compatible = [stat in supported_stats for stat in stats]
        incompatible = [i for i, x in enumerate(compatible) if not x]
        assert all(compatible), "Stat {} is not supported".format(
            stats[incompatible[0]])

        name_mapper = STATMAP

        logs = {
            f'{mode}-{name_mapper[stat]}': OrderedDict()
            for mode, stat in product(['train', 'eval'], ['cov'])
        }

        return logs, stats

    def _add_conv_layer(self, layer: torch.nn.Module):
        layer.out_features = layer.out_channels
        layer.conv_method = self.conv_method

    def get_layer_from_submodule(self, submodule: torch.nn.Module, layers: dict, name_prefix: str = ''):
            if len(submodule._modules) > 0:
                for idx, (name, subsubmodule) in enumerate(submodule._modules.items()):
                    new_prefix = name if name_prefix == '' else name_prefix+'-'+name
                    self.get_layer_from_submodule(subsubmodule, layers, new_prefix)
                return layers
            else:
                layer_name = name_prefix
                layer_type = layer_name
                if not isinstance(submodule, Conv2d) and not isinstance(submodule, Linear) and not isinstance(submodule, LSTM):
                    print(f"Skipping {layer_type}")
                    return layers
                if isinstance(submodule, Conv2d) and self.include_conv:
                    self._add_conv_layer(submodule)
                layers[layer_name] = submodule
                print('added layer {}'.format(layer_name))
                return layers

    def get_layers_recursive(self, modules: Union[list, torch.nn.Module]):
        layers = {}
        if not isinstance(modules, list) and not hasattr(
                modules, 'out_features'):
            # is a model with layers
            # check if submodule
            submodules = modules._modules  # OrderedDict
            layers = self.get_layer_from_submodule(modules, layers, '')
        else:
            for module in modules:
                layers = self.get_layer_from_submodule(module, layers, '')
        return layers

    def _get_writer(self, save_to, writers_args) -> delve.writers.AbstractWriter:
        """Create a writer to log history to `writer_dir`."""
        if issubclass(type(save_to), delve.writers.AbstractWriter):
            return save_to
        if isinstance(save_to, list):
            all_writers = []
            for saver in save_to:
                all_writers.append(self._get_writer(save_to=saver, writers_args=writers_args))
            return CompositWriter(all_writers)
        if hasattr(delve, save_to):
            writer = getattr(delve, save_to)(**writers_args)
        else:
            raise ValueError('Illegal argument for save_to "{}"'.format(save_to))
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
        return self

    def _record_stat(self, activations_batch: torch.Tensor, lstm_ae: bool, layer: torch.nn.Module, training_state: str, stat: str):
        if activations_batch.dim() == 4:  # conv layer (B x C x H x W)
            if self.conv_method == 'median':
                shape = activations_batch.shape
                reshaped_batch = activations_batch.reshape(shape[0], shape[1], shape[2] * shape[3])
                activations_batch, _ = torch.median(reshaped_batch, dim=2)  # channel median
            elif self.conv_method == 'max':
                shape = activations_batch.shape
                reshaped_batch = activations_batch.reshape(shape[0], shape[1], shape[2] * shape[3])
                activations_batch, _ = torch.max(reshaped_batch, dim=2)  # channel median
            elif self.conv_method == 'mean':
                activations_batch = torch.mean(activations_batch, dim=(2, 3))
            elif self.conv_method == 'flatten':
                activations_batch = activations_batch.view(activations_batch.size(0), -1)
            elif self.conv_method == 'channelwise':
                reshaped_batch: torch.Tensor = activations_batch.permute([1, 0, 2, 3])
                shape = reshaped_batch.shape
                reshaped_batch: torch.Tensor = reshaped_batch.flatten(1)
                reshaped_batch: torch.Tensor = reshaped_batch.permute([1, 0])
                activations_batch = reshaped_batch

        if layer.name in self.logs[f'{training_state}-{stat}']:
            self.logs[f'{training_state}-{stat}'][layer.name].update(activations_batch, lstm_ae)
        else:
            self.logs[f'{training_state}-{stat}'][layer.name] = TorchCovarianceMatrix(device=self.device)
            self.logs[f'{training_state}-{stat}'][layer.name].update(activations_batch, lstm_ae)

    def register_forward_hooks(self, layer: torch.nn.Module, stats: list):
        """Register hook to show `stats` in `layer`."""

        def record_layer_saturation(layer: torch.nn.Module, input, output):
            """Hook to register in `layer` module."""

            # Increment step counter
            layer.forward_iter += 1
            
            # VAE output is a tuple; Hence output.data throw exception
            lstm_ae = False
            if layer.name in ['encoder_lstm','encoder_output','decoder_lstm','decoder_output']:
                output = output[1][0]
                lstm_ae = True

            training_state = 'train' if layer.training else 'eval'
            if not layer.name in self.seen_samples[training_state]:
                self.seen_samples[training_state][layer.name] = 0
            if (self.max_samples is None or self.seen_samples[training_state][layer.name] < self.max_samples) and layer.forward_iter % self.log_interval == 0:
                num_samples = min(output.data.shape[0], self.max_samples-self.seen_samples[training_state][layer.name]) if self.max_samples is not None else output.data.shape[0]
                activations_batch = output.data[:num_samples]
                self.seen_samples[training_state][layer.name] += num_samples
                if self.verbose:
                    print("seen {} samples on layer {}".format(self.seen_samples[training_state][layer.name], layer.name))

                eig_vals = None

                #if 'lsat' in stats:
                #    self._record_stat(activations_batch, lstm_ae, layer, training_state, 'saturation')
                #if 'idim' in stats:
                #    self._record_stat(activations_batch, lstm_ae, layer, training_state, 'intrinsic-dimensionality')
                self._record_stat(activations_batch, lstm_ae, layer, training_state, 'covariance-matrix')

        layer.register_forward_hook(record_layer_saturation)

    def add_saturations(self, save=True):
        """
        Computes saturation and saves all stats
        :return:
        """
        for key in self.logs:
            train_sats = []
            val_sats = []
            for i, layer_name in enumerate(self.logs[key]):
                if layer_name in self.ignore_layer_names:
                   continue
                if self.logs[key][layer_name]._cov_mtx is None:
                    raise ValueError('Attempting to compute intrinsic dimensionality when covariance is not initialized')
                cov_mat = self.logs[key][layer_name].fix()
                log_values = {}
                for stat in self.stats:
                    if stat == 'lsat':
                        log_values[key.replace(STATMAP['cov'], STATMAP['lsat'])+'_'+layer_name] = compute_saturation(cov_mat, thresh=self.threshold)
                    elif stat == 'idim':
                        log_values[key.replace(STATMAP['cov'], STATMAP['idim'])+'_'+layer_name] = compute_intrinsic_dimensionality(cov_mat, thresh=self.threshold)
                    elif stat == 'cov':
                        log_values[key+'_'+layer_name] = cov_mat
                self.seen_samples[key.split('-')[0]][layer_name] = 0
                if self.reset_covariance:
                    self.logs[key][layer_name]._cov_mtx = None
                if self.layerwise_sat:
                    self.writer.add_scalars(prefix='', value_dict=log_values)

        if self.average_sat:
            self.writer.add_scalar('average-train-sat', np.mean(train_sats))
            self.writer.add_scalar('average-eval-sat', np.mean(val_sats))

        if save:
            self.save()

    def save(self):
        self.writer.save()
