import keras
import tensorflow as tf

from keras import backend as K

import numpy as np
from numpy.linalg import LinAlgError


class LayerSaturation(keras.callbacks.Callback):
    """Keras callback for computing and logging layer saturation.

    Args:
        model: Keras model
        input_data: sample input to calculate layer saturation with, eg train
        print_freq
    """
    def __init__(self, model, input_data, print_freq=1):
        self.model = model
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
            # TODO Check if has activation
            dense_outputs = self.get_layer_outputs()
            for tensor in dense_outputs:
                #                 intermediate_layer_model = Model(model.input, layer.output)
                #                 intermediate_output = intermediate_layer_model.predict(self.input_data)
                layer_name = tensor.name.split('/')[0]
                func = K.function([self.model.input] + [K.learning_phase()], [tensor])
                #                 import ipdb;ipdb.set_trace()
                intermediate_output = func([self.input_data, 0.])[0]

                tf.summary.scalar(layer_name, intermediate_output)
                self.preactivation_states[layer_name].append(intermediate_output)

    def on_epoch_end(self, epoch, logs={}):
        if epoch % 1 == 0:
            layers = self.preactivation_states.keys()
            for layer in layers:
                layer_history = self.preactivation_states[layer]
                if len(layer_history) < 2: # ?
                    continue
                history = np.stack(layer_history)[:, 0, :]  # get first representation of each batch
                history_T = history.T
                #                 import ipdb;ipdb.set_trace()
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
                weighted_sum = sum([x ** 2 for x in var_exp])
                logs[layer] = weighted_sum
                if epoch % self.print_freq == 0:
                    print(layer, weighted_sum.round(2))


class CustomTensorBoard(tf.keras.callbacks.TensorBoard):
    """Extends the TensorBoard callback to allow adding custom summaries.
      From https://groups.google.com/forum/#!topic/keras-users/rEJ1xYqD3AM.

    Args:
        user_defined_freq: frequency (in epochs) at which to compute summaries
            defined by the user by calling tf.summary in the model code. If set to
            0, user-defined summaries won't be computed. Validation data must be
            specified for summary visualization.
        kwargs: Passed to tf.keras.callbacks.TensorBoard.
    """

    def __init__(self, user_defined_freq=0, **kwargs):
        self.user_defined_freq = user_defined_freq
        super(CustomTensorBoard, self).__init__(**kwargs)

    def on_epoch_begin(self, epoch, logs=None):
        """Add user-def. op to Model eval_function callbacks, reset batch count."""

        # check if histogram summary should be run for this epoch
        if self.user_defined_freq and epoch % self.user_defined_freq == 0:
            self._epoch = epoch
            # pylint: disable=protected-access
            # add the user-defined summary ops if it should run this epoch
            self.model._make_eval_function()
            if self.merged not in self.model._eval_function.fetches:
                self.model._eval_function.fetches.append(self.merged)
                self.model._eval_function.fetch_callbacks[
                    self.merged] = self._fetch_callback
            # pylint: enable=protected-access

        super(CustomTensorBoard, self).on_epoch_begin(epoch, logs=None)

    def on_epoch_end(self, epoch, logs=None):
        """Checks if summary ops should run next epoch, logs scalar summaries."""

        # pop the user-defined summary op after each epoch
        if self.user_defined_freq:
            # pylint: disable=protected-access
            if self.merged in self.model._eval_function.fetches:
                self.model._eval_function.fetches.remove(self.merged)
            if self.merged in self.model._eval_function.fetch_callbacks:
                self.model._eval_function.fetch_callbacks.pop(self.merged)
            # pylint: enable=protected-access

        super(CustomTensorBoard, self).on_epoch_end(epoch, logs=logs)
