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
np_array = np.array([[1, 2], [3, 4]], dtype=float)
mojo_tensor = ndarray_to_tensor[DType.float64](np_array)
```

# Installation
Add `https://repo.prefix.dev/modular-community` to the channels section of your of your
`mojoproject.toml` file.
Then run:
```bash
magic add bridge
```

# Dependencies
Requires numpy and mojo.

# Contributing
Please refer to the [contribution guidelines](https://gitlab.com/hylkedonker/bridge/-/blob/main/CONTRIBUTING.md) before contributing.

# License
This code is licensed under the terms of the [MIT License](LICENSE.txt).