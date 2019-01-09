import keras
import keras.backend as K
from keras.models import Sequential, Model
import numpy as np

from numpy.linalg import LinAlgError

class LayerSaturation(keras.callbacks.Callback):
    """Calculates and outputs dense layer saturation for Keras models.
    Args:
        input_data (array) : input data (batch_size x feature_size)
        print_freq (int)   : Frequency in number of epochs to print out saturation
    """

    def __init__(self, input_data, saturation_metric = 'simpson_di', sample_rate=10, print_freq=1):
        self.input_data = input_data
        self.saturation_metric = saturation_metric
        self.sample_rate = sample_rate
        self.print_freq = print_freq

    def get_layer_outputs(self):
        dense_outputs = []
        for layer in self.model.layers[1:]:
            if 'dense' in layer.name:
                if layer.activation.__name__ is not 'linear':
                    preactivation_tensor = layer.output.op.inputs[0]
                else:
                    preactivation_tensor = layer.output
                dense_outputs.append(preactivation_tensor)
        return dense_outputs

    def on_train_begin(self, logs={}):
        self.preactivation_states = {}
        dense_outputs = self.get_layer_outputs()
        for tensor in dense_outputs:
            layer_name = tensor.name.split('/')[0]
            self.preactivation_states[layer_name] = []

    def on_batch_end(self, batch, logs={}):
        if batch % self.sample_rate == 0:
            dense_outputs = self.get_layer_outputs()
            for tensor in dense_outputs:
                layer_name = tensor.name.split('/')[0]
                func = K.function([self.model.input] + [K.learning_phase()], [tensor])
                intermediate_output = func([self.input_data, 1.])[0]
                self.preactivation_states[layer_name].append(intermediate_output)

    def on_epoch_end(self, epoch, logs={}):
        if epoch % 1 == 0:
            layers = self.preactivation_states.keys()
            for layer in layers:
                layer_history = self.preactivation_states[layer]
                if len(layer_history) < 2:
                    continue
                history = np.stack(layer_history)[:, 0, :] # get first representation of each batch
                history_T = history.T
                try:
                    cov = np.cov(history_T)
                except LinAlgError:
                    continue
                eig_vals, eig_vecs = np.linalg.eigh(cov)

                # Make a list of (eigenvalue, eigenvector) tuples
                eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
                # Sort the (eigenvalue, eigenvector) tuples from high to low
                eig_pairs = sorted(eig_pairs, key=lambda x: x[0], reverse=True)
                eig_vals, eig_vecs = zip(*eig_pairs)
                tot = sum(eig_vals)
                var_exp = [(i / tot) for i in eig_vals]
                if self.saturation_metric == 'simpson_di':
                    weighted_sum = sum([x ** 2 for x in var_exp])
                    logs[layer] = weighted_sum
                    if epoch % self.print_freq == 0:
                        print(layer, weighted_sum.round(2))
                elif self.saturation_metric == 'cumulative_99':
                    eig_vals = np.array(eig_vals)
                    cumsum = eig_vals.cumsum()
                    total_variance_explained = cumsum / eig_vals.sum()
                    K = np.argmax(total_variance_explained > 0.99) + 1
                    saturation = K / len(eig_vals)
                    logs[layer] = saturation
                    if epoch % self.print_freq == 0:
                        print(layer, saturation.round(2))
                else:
                    raise NotImplementedError("{} not yet implemented".format(self.saturation_metric))

class TensorBoardSat(keras.callbacks.TensorBoard):
    def __init__(self, **kwargs):
        super(TensorBoardSat, self).__init__(**kwargs)
        self.dtw_image_summary = None

    def on_epoch_end(self, epoch, logs={}):
        super(TensorBoardSat).on_train_begin(self, logs=logs)
        if epoch % 1 == 0:
            self.writer.add_summary("Saturation", 1)
