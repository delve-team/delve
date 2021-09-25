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
    def record_stat(self, layer: torch.nn.Module, name: str, logs: Dict[str, Dict[str, Union[Optional[TorchCovarianceMatrix]]]],
                    *args, **kwargs) -> None:
        """Record the stat of a specific layer using the forward hook interface.

        Args:
            layer:  the layer to record
            name:   the name of the layer
            logs:   the logging dictionary
            args:   any additional arguments
            kwargs: any additional keyword-arguments

        """
        ...