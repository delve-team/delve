from abc import ABC, abstractmethod
from typing import Dict, Union, Optional

import torch

from delve import TorchCovarianceMatrix


class AbstractLayerHandler(ABC):

    @abstractmethod
    def can_handle(self, layer: torch.nn.Module) -> bool:
        """This function checks if a layer can be handled by delve

        Args:
            layer: the layer in question

        Returns:
            True if the layer can be handled by the handler
        """
        ...

    def preprocess(self, layer: torch.nn.Module, name: str) -> torch.nn.Module:
        """Preprocesses the layer to prepare it for layer recording

        Args:
            layer:  the layer in question
            name:   the name of the layer

        Returns:
            the preprocessed model. Modification happen inplace.
        """
        return layer


    @abstractmethod
    def record_stat(self, activations_batch: torch.Tensor, lstm_ae: bool,
                    layer: torch.nn.Module, training_state: str, stat: str,
                    logs: Dict[str, Dict[str, Union[Optional[TorchCovarianceMatrix]]]],
                    *args, **kwargs):
        """Record the stat of a specific layer using the forward hook interface.

                Args:
                    activations_batch:  the layer output
                    layer:              the layer as PyTorch-module
                    training_state:     the current state of training, may be 'train' or 'eval'
                    stat:               the statistic to compute (for example "sat")
                    logs:               the logs containing the TorchCovarianceMatrices
                    *args:              any additional arguments
                    **kwargs:           any additional keyword arguments

                """
        ...
    

class Conv2DHandler(AbstractLayerHandler):

    def preprocess(self, layer: torch.nn.Module, name: str) -> torch.nn.Module:
        ...

    def record_stat(self, activations_batch: torch.Tensor, lstm_ae: bool, layer: torch.nn.Module, training_state: str,
                    stat: str, logs: Dict[str, Dict[str, Union[Optional[TorchCovarianceMatrix]]]], *args, **kwargs):
        pass

    def can_handle(self, layer: torch.nn.Module) -> bool:
        pass