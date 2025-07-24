# Bridge
[![SAST](https://gitlab.com/hylkedonker/bridge/badges/main/pipeline.svg?job=sast)](https://gitlab.com/hylkedonker/bridge/-/security/dashboard)
<p align="center">
  <img src="misc/image.jpeg" width="30%" alt="Bridge">
</p>

This package acts as a bridge between Mojo native types and those from the Python world.

Example usage, to convert a NumPy array to a Mojo `LayoutTensor`:
```mojo
from python import Python
from bridge.numpy import ndarray_to_tensor

var np = Python.import_module("numpy")
np_array = np.arange(6.0).reshape(2,3)

# Convert to new-style `LayoutTensor`.
mojo_tensor = ndarray_to_tensor[order=2](np_array)
```
Or to achieve the reverse:
```mojo
from bridge.numpy import tensor_to_ndarray

# Convert `LayoutTensor` to numpy array.
np_array = tensor_to_ndarray(mojo_tensor)
```

# Installation
Add the `modular-community` channel to  your `mojoproject.toml` file and `bridge` to
your dependencies, by running:
```bash
pixi project channel add "https://repo.prefix.dev/modular-community"
pixi add bridge
```
That's it, success! 🎉

# Dependencies
Requires numpy and mojo.

# Contributing
Please refer to the [contribution guidelines](https://gitlab.com/hylkedonker/bridge/-/blob/main/CONTRIBUTING.md) before contributing.

# License
This code is licensed under the terms of the [MIT License](LICENSE.txt).