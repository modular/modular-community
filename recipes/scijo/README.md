# SciJo

SciJo is a scientific computing library written in Mojo and inspired by SciPy.
It builds on [NuMojo](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo)
and provides numerical integration, differentiation, interpolation, FFTs,
linear algebra, optimization, root finding, and physical constants.

## Installation

Add the Modular community channel and SciJo to your Pixi project:

```toml
[workspace]
channels = [
  "https://conda.modular.com/max",
  "conda-forge",
  "https://repo.prefix.dev/modular-community",
]

[dependencies]
scijo = "=0.3.0"
```

SciJo 0.3.0 requires Mojo 1.1 and NuMojo 0.11.

See the [project README](https://github.com/mojomath/SciJo#readme) and
[changelog](https://github.com/mojomath/SciJo/blob/main/docs/changelog.md) for
usage examples and release details.
