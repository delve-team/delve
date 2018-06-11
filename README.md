# delve

Looking inside the black box of deep learning with spectral analysis.


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

Use delve to compare models with varying length layers (here 5, 8 and 10):

![saturation-screenshot]('images/saturation.png')

### Spectral analysis

Plot the top eigenvalues of each layer:

```python
stats = CheckLayerSat('runs', layers, 'spectrum')
```
