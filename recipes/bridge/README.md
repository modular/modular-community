# Bridge
[![SAST](https://gitlab.com/hylkedonker/bridge/badges/main/pipeline.svg?job=sast)](https://gitlab.com/hylkedonker/bridge/-/security/dashboard)
<p align="center">
  <img src="misc/image.jpeg" width="30%" alt="Bridge">
</p>

This package acts as a bridge between Mojo native types and those from the Python world.

Example usage, to convert a NumPy array to a Mojo tensor:
```mojo
from python import Python
from bridge.numpy import ndarray_to_tensor

var np = Python.import_module("numpy")
np_array = np.arange(6.0).reshape(2,3)
mojo_tensor = ndarray_to_tensor[DType.float64](np_array)
```
Or to achieve the reverse:
```mojo
from tensor import Tensor

from bridge.numpy import tensor_to_ndarray

values = List[Float64](0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
mojo_tensor = Tensor[DType.float64](shape=(2, 3), list=values)
np_array = tensor_to_ndarray(mojo_tensor)
```

# Installation
Add the `modular-community` channel to  your `mojoproject.toml` file and `bridge` to
your dependencies, by running:
```bash
magic project channel add "https://repo.prefix.dev/modular-community"
magic add bridge
```
That's it, success! 🎉

# Dependencies
Requires numpy and mojo.

# Contributing
Please refer to the [contribution guidelines](https://gitlab.com/hylkedonker/bridge/-/blob/main/CONTRIBUTING.md) before contributing.

# License
This code is licensed under the terms of the [MIT License](LICENSE.txt).