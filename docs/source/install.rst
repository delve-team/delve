Installation
============

Installing Delve
----------------

Delve require Python 3.6+ to be installed.

To install via pip:

.. code::

   pip install delve

To install the latest development version, clone the `GitHub` repository and use the setup script::

   git clone https://github.com/delve-team/delve.git
   cd delve
   pip install .

Usage
-----

Instantiate the :class:`~delve.torchcallback.SaturationTracker` class where you define your PyTorch training loop, as in the example::

   from torch import nn
   from delve import SaturationTracker

   ...

   model = nn.ModuleDict({
                 'conv1': nn.Conv2d(1, 8, 3, padding=1),
                 'linear1': nn.Linear(3, 1),
   })


   layers = [model.conv1, model.linear1]
   stats = SaturationTracker('regression/h{}'.format(h),
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

This will hook into the layers in ``layers`` and log the statistics, in this case ``lsat`` (layer saturation). It will save images to ``regression``.
