Examples
========

Delve allows the user to create plots and log records in various formats.

Plotting
--------

Delve allows plotting results every epoch using ``save_to="csvplot"``, which will create automated plots from the metrics
recorded in the ``stats`` argument. The plots depict the layers generally in order of the forward pass.

.. image:: gallery/images/VGG16-Cifar10-r32-bs256-e90idim_epoch_88.png
  :width: 400
  :alt: Automatically generated plot of intrinsic dimensionality computed on the training set of Cifar10 on  VGG16 at the 90th epoch of training.

.. image:: gallery/images/VGG16-Cifar10-r32-bs256-e90lsat_epoch_88.png
  :width: 400
  :alt: Automatically generated plot of saturation computed on the training set of Cifar10 on  VGG16 at the 90th epoch of training.


Logging
-------

Delve logs results with the ``logging`` package and shows progress with ``tqdm``.
