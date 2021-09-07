.. Delve documentation master file, created by
   sphinx-quickstart on Sun Aug 22 13:20:36 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Delve Documentation
===================

Delve is library for analyzing eigenspaces of neural networks.

Delve was developed to help researchers identify information flow through a layer.
Specifically, it computes layer `saturation`, a measure of the covariance of features in a layer.

If you use Delve in your publication, please cite:

.. code-block:: txt

   @software{delve,
   author       = {Justin Shenk and
                     Mats L. Richter and
                     Wolf Byttner and
                     Michał Marcinkiewicz},
   title        = {delve-team/delve: v0.1.45},
   month        = aug,
   year         = 2021,
   publisher    = {Zenodo},
   version      = {v0.1.45},
   doi          = {10.5281/zenodo.5233860},
   url          = {https://doi.org/10.5281/zenodo.5233860}
   }
    
Delve allows extracting features from neural network layers and computing the eigenspace of several layers.

Supported Layers
----------------

* Dense/Linear
* LSTM
* Convolutional

Statistics
----------

Layer eigenspace computation allows computing information flow between layers, including:

* feature variance
* feature covariance 
* layer feature instrinsic dimensionality

Several statistics are supported:

.. code::

   idim        : intrinsic dimensionality
   lsat        : layer saturation (intrinsic dimensionality divided by feature space dimensionality)
   cov         : the covariance-matrix (only saveable using the 'npy' save strategy)
   det         : the determinant of the covariance matrix (also known as generalized variance)
   trc         : the trace of the covariance matrix, generally a more useful metric than det for determining
               the total variance of the data than the determinant.
               However note that this does not take the correlation between
               features into account. On the other hand, in most cases the determinent will be zero, since
               there will be very strongly correlated features, so trace might be the better option.
   dtrc        : the trace of the diagonalmatrix, another way of measuring the dispersion of the data.
   lsat        : layer saturation (intrinsic dimensionality
               divided by feature space dimensionality)
   cov         : the covariance-matrix (only saveable using
               the 'npy' save strategy)
   embed       : samples embedded in the eigenspace of dimension 2


To support researchers, it allows saving plots at various intervals through the :class:`~delve.torchcallback.CheckLayerSat` class.


Getting Started
---------------

Install with

.. code::

   pip install delve

then instantiate the :class:`~delve.torchcallback.CheckLayerSat` class, as in the example::

   model = nn.ModuleDict({
                 'conv1': nn.Conv2d(1, 8, 3, padding=1),
                 'linear1': nn.Linear(3, 1),
   })


   layers = [model.conv1, model.linear1]
   stats = CheckLayerSat('regression/h{}'.format(h),
      save_to="plotcsv",
      modules=layers,      
      stats=["lsat"]
   )

   ...

   for _ in range(10):
      y_pred = model(x)
      loss = loss_fn(y_pred, y)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      stats.add_saturations()
   
   stats.close()

This will hook into the layers in ``layers`` and log the statistics, in this case ``lsat`` (layer saturation).

.. toctree::
   :maxdepth: 1
   :caption: User Guide
   
   Saturation <saturation>
   Example Plots <examples>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
