# SciJo

<div align="center">
    <img src="./scijo.png" alt="SciJo Logo" width="200" style="border-radius: 32px; margin-bottom: 200px; display: block; border: 3px solid rgba(0, 0, 0, 0.15); box-shadow: 0 20px 45px rgba(0, 0, 0, 0.25); background: #fff;"/>
  <p style="font-size: 1.2em; color: #666; margin: 0; padding: 10px 20px; line-height: 1.5;">
    <em>SciJo is a high-performance scientific computing library written in Mojo🔥, inspired by SciPy.</em>
  </p>
</div>

**[Manual»](docs/MANUAL.md)** | **[Changelog»](docs/changelog.md)** | **[Examples»](examples/)**

## Overview

SciJo is a high-performance scientific computing library for Mojo that brings the power and familiarity of SciPy to the Mojo ecosystem. Written in pure Mojo and built on top of **[NuMojo](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo)**, SciJo combines the performance benefits of native compilation with the type safety guarantees of Mojo's advanced type system.

## Features

- **Pure Mojo**: Native implementation for maximum performance
- **Familiar APIs**: SciPy-inspired interfaces for easy adoption
- **Type Safe**: Compile-time guarantees with Mojo's type system
- **NuMojo Backend**: Efficient array operations and complex number support

## Current Modules

### Numerical Differentiation (`scijo.differentiate`)
Accurate derivatives using finite difference methods:
- **`derivative`**: Central, forward, and backward differences
- **`jacobian`**: Jacobian matrix computation for vector-valued functions
- **Order control**: Specify accuracy order (1-6 for forward/backward, 2-8 for central)
- **Adaptive stepping**: Automatic step size refinement with Richardson extrapolation
- **Error estimation**: Built-in convergence tracking

### Integration (`scijo.integrate`)
Numerical integration with adaptive algorithms:
- **`quad`**: Adaptive quadrature via MSL/QUADPACK
  - `method="qng"` - non-adaptive Gauss-Kronrod-Patterson (10, 21, 43, 87 point rules)
  - `method="qag"` - adaptive Gauss-Kronrod with configurable rule (`qag_rule=`)
  - `method="qags"` - adaptive + Wynn epsilon extrapolation
  - Rule constants: `QAG_GK15`, `QAG_GK21` (default), `QAG_GK31`, `QAG_GK41`, `QAG_GK51`, `QAG_GK61`
- **`trapezoid`**: Composite trapezoidal rule for uniform or non-uniform grids
- **`simpson`**: Simpson's rule for discrete data
- **`romb`**: Romberg integration with Richardson extrapolation

### Interpolation (`scijo.interpolate`)
1D data interpolation:
- **`LinearInterpolator`** / **`interp1d`**: Linear interpolation, callable object and functional interface
- **`CubicSpline`**: Natural cubic spline interpolator (`scipy.interpolate.CubicSpline`-compatible)
- **`Akima1DInterpolator`**: Akima piecewise cubic interpolator (`scipy.interpolate.Akima1DInterpolator`-compatible)
- **`interp`**: Functional interface supporting `type="linear"`, `"cubic"`, `"akima"`
- Out-of-bounds control via `bounds_error` / `fill_value`
- Compatible with NuMojo arrays

### FFT (`scijo.fft`)
Fast Fourier Transform operations:
- **`fft`**: Forward FFT using Cooley-Tukey algorithm
- **`ifft`**: Inverse FFT with proper normalization
- Supports complex arrays (power-of-2 sizes)
- Compatible with NumPy's FFT conventions

### Physical Constants (`scijo.constants`)
Access fundamental physical constants from CODATA 2022:
- Comprehensive physical constants with values, units, and uncertainties
- Mathematical constants (pi, golden ratio, etc.)
- SI prefixes, binary prefixes, and unit conversions
- Helper functions: `value()`, `unit()`, `precision()`, `find()`
- Temperature conversion utilities
- Compatible with `scipy.constants` structure

### Optimization (`scijo.optimize`)
Scalar root-finding and minimization:
- **`root_scalar`**: Unified interface for root finding
  - **`newton`**: Newton-Raphson method
  - **`bisect`**: Bisection method
  - **`secant`**: Secant method (derivative-free)
  - **`brent`**: Brent's bracketed method (MSL backend)
- **`minimize_scalar`**: Scalar function minimization
  - Brent's method, golden section search, bounded minimization

## Installation

### Method 1: Git Installation via pixi-build-backend
Install SciJo directly from the GitHub repository to access the latest features.

Add to your `pixi.toml`:
```toml
[workspace]
preview = ["pixi-build"]

[dependencies]
mojo = ">=1.0.0,<2"
scijo = { git = "https://github.com/mojomath/SciJo.git", branch = "main"}
```

Then run:
```bash
pixi install
```

### Method 2: Stable Release via Pixi (prefix.dev)
For most users, we recommend installing a stable release through Pixi for guaranteed compatibility and reproducibility.

Add the following to your `pixi.toml`:
```toml
[workspace]
channels = ["https://repo.prefix.dev/modular-community"]
```

Then run:
```bash
pixi add scijo
```

### Method 3: Build from Source
```bash
# Clone and build
git clone https://github.com/mojomath/SciJo.git
cd SciJo
pixi run package   # mojo precompile scijo && cp scijo.mojoc tests/

# Point your program at the precompiled package
mojo run -I /path/to/SciJo -I /path/to/SciJo/tests my_program.mojo
```

## Quick Start

### Numerical Differentiation
```mojo
from scijo.differentiate import derivative
from scijo.prelude import *

def simple_function[
    dtype: DType
](x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]) capturing -> Scalar[dtype]:
    var a = args.value()[0]
    return a * x * x + 2.0 * x + 1.0

def main() raises:
    var args: List[Scalar[f64]] = [2.0]
    var result = derivative[f64, simple_function, step_direction=0](
        x0=1.0,
        args=args^,
        atol=1e-8,
        rtol=1e-8,
        order=6,
    )
    print("Derivative result:", result.df)
```

### Integration
```mojo
from scijo.integrate import quad
from scijo.prelude import *

def simple_function[
    dtype: DType
](x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]) capturing -> Scalar[
    dtype
]:
    """A simple function for testing."""
    var a = args.value()[0]
    return a * x * x + 2.0 * x + 1.0

def main() raises:
    var args: List[Scalar[f64]] = [2.0]
    var result = quad[f64, simple_function](
        a=0.0,
        b=1.0,
        args=args^,
        atol=1e-6,
        rtol=1e-6,
    )
    print("Integral value:", result.integral)
```

### Interpolation
```mojo
from scijo.interpolate import interp1d, CubicSpline, Akima1DInterpolator
import numojo as nm

def main() raises:
    var x = nm.linspace[nm.f64](0.0, 10.0, 11)
    var y = x * x

    # Linear (callable object)
    var li = interp1d(x, y, bounds_error=False)
    print(li(Scalar[nm.f64](3.7)))

    # Natural cubic spline
    var cs = CubicSpline(x, y)
    print(cs(nm.linspace[nm.f64](0.5, 9.5, 5)))

    # Akima
    var ak = Akima1DInterpolator(x, y)
    print(ak(Scalar[nm.f64](3.7)))
```

### FFT
```mojo
from scijo.fft import fft, ifft
import numojo as nm

def main() raises:
    # Create complex array
    var arr = nm.arange[nm.cf64](nm.CScalar[nm.cf64](0), nm.CScalar[nm.cf64](8))

    # Forward FFT
    var y_fft = fft[nm.cf64](arr)
    print("FFT result:", y_fft)

    # Inverse FFT
    var y_ifft = ifft[nm.cf64](y_fft)
    print("IFFT result:", y_ifft)
```

### Physical Constants
```mojo
from scijo.constants import physical_constants, value, unit

def main() raises:
    print("Speed of light:", value("speed_of_light_in_vacuum"), "m/s")
    print("Planck constant:", value("Planck_constant"), unit("Planck_constant"))
```

### Optimization
```mojo
from scijo.optimize import root_scalar, minimize_scalar
from scijo.prelude import *

def f[
    dtype: DType
](x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]) capturing -> Scalar[dtype]:
    return x * x - 2.0

def objective[
    dtype: DType
](x: Scalar[dtype], args: Optional[List[Scalar[dtype]]]) capturing -> Scalar[dtype]:
    return (x - 2.0) * (x - 2.0) + 1.0

def main() raises:
    # Root finding - `method` is a compile-time parameter, so it goes in the brackets
    var root = root_scalar[f64, f, method="bisect"](bracket=(1.0, 2.0))
    print("Root:", root.root)

    # Minimization
    var result = minimize_scalar[f64, objective, method="Brent"](
        bracket=(0.0, 4.0),
        atol=1e-8,
        maxiter=100,
    )
    print("Minimum at:", result.x)
```

See the **[Manual»](docs/MANUAL.md)** for a full prose tour of every module, and
**[examples/»](examples/)** for runnable, longer versions of the snippets above
(run them all with `examples/run_all.sh`, after `pixi run package`).

## Roadmap

### Near Term
- 2D FFT support
- Multi-dimensional root finding and optimization
- Expand differentiation module (higher-order Jacobian, Hessian)

### Future
- **Optimization**: Multi-dimensional minimization, curve fitting
- **Signal Processing**: Filtering, windowing, convolution
- **Linear Algebra**: Matrix decompositions (SVD, QR, Cholesky)
- **Sparse Matrices**: Efficient storage and operations

## Contributing

Contributions are most welcome! Feel free to add a functionality and open a PR!

Priority areas:
- Algorithm implementations (see Roadmap)
- Performance benchmarks and optimization
- Tests and documentation
- Bug reports and feature requests

## License

Distributed under the Apache 2.0 License with LLVM Exceptions. See [LICENSE](LICENSE) for more information.

## Citation
Feel free to cite SciJo in your work, helps with visibility :)
```bibtex
@software{scijo,
  author = {Shivasankar K.A. and SciJo Contributors},
  title = {SciJo: High-Performance Scientific Computing in Mojo},
  url = {https://github.com/mojomath/SciJo},
  year = {2026}
}
```

---

⚠️ **Note**: This library is in early development and may introduce breaking changes between versions.
