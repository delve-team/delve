import keras
import keras.backend as K
from keras.models import Sequential, Model
import numpy as np
from sklearn.decomposition import PCA


class LayerSaturation(keras.callbacks.Callback):
    """Calculates and outputs dense layer saturation for Keras models.
    Args:
        input_data (array) : input data (batch_size x feature_size)
        print_freq (int)   : Frequency in number of epochs to print out saturation

            supported stats are:
                lsat       : layer saturation
                bcov       : batch covariance
                eigendist  : eigenvalue distribution
                neigendist : normalized eigenvalue distribution
                spectrum   : top-N eigenvalues of covariance matrix
                spectral   : spectral analysis (eigendist, neigendist, and spectrum)

        verbose (bool)     : print saturation for every layer during training
    """

    def __init__(self, input_data, print_freq=1):
        self.input_data = input_data
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
        if batch % 10 == 0:
            dense_outputs = self.get_layer_outputs()
            for tensor in dense_outputs:
                layer_name = tensor.name.split('/')[0]
                func = K.function([self.model.input] + [K.learning_phase()], [tensor])
                intermediate_output = func([self.input_data, 0.])[0]
                self.preactivation_states[layer_name].append(intermediate_output)

    def on_epoch_end(self, epoch, logs={}):
        if epoch % 1 == 0:
            layers = self.preactivation_states.keys()
            for layer in layers:
                layer_history = self.preactivation_states[layer]
                pca = PCA()
                history = np.stack(layer_history)[:, 0, :]  # get first representation of each batch
                history_t = history.T
                pca.fit(history_t)
                weighted_sum = sum([x ** 2 for x in pca.explained_variance_ratio_])
                logs[layer] = weighted_sum
                if epoch % self.print_freq == 0:
                    print(layer, weighted_sum.round(2))
