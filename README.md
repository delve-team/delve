# delve

Inspect layer saturation and spectral data in your PyTorch models.

**delve**, *verb*:
```
1. Reach inside a receptacle and search for something
2. Research or make painstaking inquiries into something
```
## Getting started

```bash
pip install delve
```

### Layer Saturation
Pass a PyTorch model (or layers) to CheckLayerSat:

```python
from delve import CheckLayerSat

model = TwoLayerNet()
layers = [model.linear1, model.linear2]
stats = CheckLayerSat('runs', layers)
```

#### Optimize neural network topology

Ever wonder how big your layer size should be? Delve helps you visualize the effect of modifying the layer size on your layer saturation.

For example, see how modifying the hidden layer size of this network affects the second layer saturation but not the first. Here we show variations of the fully-connected "linear2" layer (blue is 256 and orange is 8):

![saturation](images/layer1-saturation.png)

![saturation](images/layer2-saturation.png)

### Spectral analysis

Plot the top 5 eigenvalues of each layer:

```python
stats = CheckLayerSat('runs', layers, 'spectrum')
```

![spectrum](images/spectrum.png)

### Intrinsic dimensionality

View the intrinsic dimensionality of models in realtime:


![intrinsic_dimensionality-layer2](images/layer2-intrinsic.png)

This comparison suggests that the 8-unit layer (light blue) is too saturated and that a larger layer is needed.
