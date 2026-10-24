# SciJo

<div align="center">
    <img src="./assets/scijo.png" alt="SciJo Logo" width="200" style="border-radius: 32px; margin-bottom: 200px; display: block; border: 3px solid rgba(0, 0, 0, 0.15); box-shadow: 0 20px 45px rgba(0, 0, 0, 0.25); background: #fff;"/>
  <p style="font-size: 1.2em; color: #666; margin: 0; padding: 10px 20px; line-height: 1.5;">
    <em>SciJo is a high-performance scientific computing library written in Mojo🔥, inspired by SciPy.</em>
  </p>
</div>

**[Manual»](https://github.com/mojomath/SciJo/blob/main/docs/MANUAL.md)** | **[Changelog»](https://github.com/mojomath/SciJo/blob/main/docs/changelog.md)** | **[Examples»](https://github.com/mojomath/SciJo/tree/main/examples)**

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
- **`hessian`**: Hessian matrix for scalar-valued functions, or Hessian tensor (one matrix per output component) for vector-valued functions
- **Order control**: Specify accuracy order (1-6 for forward/backward, 2-8 for central)
- **Adaptive stepping**: Automatic step size refinement with Richardson extrapolation
- **Error estimation**: Built-in convergence tracking

### Integration (`scijo.integrate`)
Numerical integration with adaptive algorithms:
- **`quad`**: Adaptive quadrature via native Gauss-Kronrod rules (QUADPACK-style)
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
- **`fft`** / **`ifft`**: Forward/inverse FFT using the Cooley-Tukey algorithm
- **`fftn`** / **`ifftn`**: N-dimensional forward/inverse FFT over one or more axes
- **`fft2`** / **`ifft2`**: 2-dimensional forward/inverse FFT (convenience wrapper over `fftn`/`ifftn`)
- **`rfft`** / **`irfft`**: Real-input FFT and its inverse
- Supports complex arrays (power-of-2 sizes per transformed axis)
- Compatible with NumPy's FFT conventions

### Physical Constants (`scijo.constants`)
Access fundamental physical constants from CODATA 2022:
- Comprehensive physical constants with values, units, and uncertainties
- Mathematical constants (pi, golden ratio, etc.)
- SI prefixes, binary prefixes, and unit conversions
- Helper functions: `value()`, `unit()`, `precision()`, `find()`
- Temperature conversion utilities
- Compatible with `scipy.constants` structure

### Linear Algebra (`scijo.linalg`)
Linear algebra, built on NuMojo's linalg primitives:
- **Decompositions**: `lu` (pivoted, `A == P @ L @ U`), `qr` (reduced, Householder), `cholesky` (lower/upper)
- **Eigenvalues/SVD**: `eigh`/`eigvalsh` (symmetric, via cyclic Jacobi), `svd`/`svdvals` (via one-sided Jacobi), `matrix_rank`, `cond`
- **Solvers**: `solve`, `solve_triangular`, `inv`, `pinv` (SVD-based, rank-deficient OK), `lstsq` — all accept 1-D or 2-D right-hand sides
- **Norms**: `norm` (vector `1`/`2`/`inf`, matrix `fro`/`1`/`2`/`inf`), `det`, `trace`
- Re-exports `matmul`, `dot`, `outer`, `kron`, `tensordot`, `cross`, `diagonal` from NuMojo for convenience
- The general (non-symmetric) eigenvalue problem (`eig`/`eigvals`, possibly complex eigenvalues) is not yet available

### Optimization (`scijo.optimize`)
Scalar and multi-dimensional root-finding, and scalar and multi-dimensional
minimization:
- **`root_scalar`**: Unified interface for scalar root finding
  - **`newton`**: Newton-Raphson method
  - **`bisect`**: Bisection method
  - **`secant`**: Secant method (derivative-free)
  - **`brent`**: Brent's bracketed method
- **`root`**: Multi-dimensional root finding (`f: R^n -> R^n`)
  - `method="newton"` (default): finite-difference Jacobian, or supply an analytic one: `root[f64, f, jac](x0)`
  - `method="broyden1"`: Broyden's quasi-Newton method, derivative-free (never forms a Jacobian)
- **`minimize_scalar`**: Scalar function minimization
  - Brent's method, golden section search, bounded minimization
- **`minimize`**: Multi-dimensional minimization (`f: R^n -> R`)
  - `method="Nelder-Mead"` (default): derivative-free simplex method
  - `method="BFGS"` / `method="CG"`: gradient-based, with the gradient estimated via finite differences

## Installation

### Method 1: Git Installation with pixi-build-mojo
Install the SciJo v0.3.0 release directly from GitHub.

Add to your `pixi.toml`:
```toml
[workspace]
channels = [
    "https://conda.modular.com/max",
    "conda-forge",
    "https://repo.prefix.dev/modular-community",
]
platforms = ["osx-arm64", "linux-64"]
preview = ["pixi-build"]

[dependencies]
scijo = { git = "https://github.com/mojomath/SciJo.git", tag = "v0.3.0" }
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
channels = [
    "https://conda.modular.com/max",
    "conda-forge",
    "https://repo.prefix.dev/modular-community",
]

[dependencies]
scijo = "=0.3.0"
```

Then run:
```bash
pixi install
```

SciJo v0.3.0 requires Mojo 1.1 and NuMojo 0.11.

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
](x: Scalar[dtype]) capturing -> Scalar[dtype]:
    return 2.0 * x * x + 2.0 * x + 1.0

def main() raises:
    var result = derivative[f64, simple_function, step_direction=0](
        x0=1.0,
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
](x: Scalar[dtype]) capturing -> Scalar[dtype]:
    """A simple function for testing."""
    return 2.0 * x * x + 2.0 * x + 1.0

def main() raises:
    var result = quad[f64, simple_function](
        a=0.0,
        b=1.0,
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
from scijo.fft import fft, ifft, fftn, ifftn
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

    # N-dimensional FFT: transforms over every axis by default
    var arr2d = nm.arange[nm.cf64](
        nm.CScalar[nm.cf64](0), nm.CScalar[nm.cf64](16)
    ).reshape(nm.Shape(4, 4))
    var y_fftn = fftn[nm.cf64](arr2d)
    print("FFTN result:", y_fftn)
    var y_ifftn = ifftn[nm.cf64](y_fftn)
    print("IFFTN result:", y_ifftn)
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
](x: Scalar[dtype]) capturing -> Scalar[dtype]:
    return x * x - 2.0

def objective[
    dtype: DType
](x: Scalar[dtype]) capturing -> Scalar[dtype]:
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

### Multi-Dimensional Root Finding
```mojo
from scijo.optimize import root
from scijo.linalg import matmul
import numojo as nm

def f[dtype: DType](x: nm.NDArray[dtype]) capturing raises -> nm.NDArray[dtype]:
    var A = nm.fromstring[dtype]("[[3, 1], [1, 2]]")
    var b = nm.fromstring[dtype]("[9, 8]")
    return matmul(A, x) - b

def main() raises:
    var x0 = nm.fromstring[nm.f64]("[0, 0]")
    var result = root[nm.f64, f](x0)
    print("Root:", result.x)
```

### Multi-Dimensional Minimization
```mojo
from scijo.optimize import minimize
import numojo as nm

def rosenbrock[
    dtype: DType
](x: nm.NDArray[dtype]) capturing raises -> Scalar[dtype]:
    var a = 1.0 - x.item(0)
    var b = x.item(1) - x.item(0) * x.item(0)
    return a * a + 100.0 * b * b

def main() raises:
    var x0 = nm.fromstring[nm.f64]("[-1.2, 1.0]")
    var result = minimize[nm.f64, rosenbrock](x0, maxiter=2000)
    print("Minimum at:", result.x)

    # Or a gradient-based method (gradient estimated via finite differences):
    var result_bfgs = minimize[nm.f64, rosenbrock, method="BFGS"](x0)
    print("Minimum at:", result_bfgs.x)
```

### Linear Algebra
```mojo
from scijo.linalg import eigh, solve, lu, matmul
import numojo as nm

def main() raises:
    var A = nm.fromstring[nm.f64]("[[3, 1], [1, 2]]")
    var b = nm.fromstring[nm.f64]("[9, 8]")
    print("x:", solve(A, b))

    var PLU = lu(A)
    var P = PLU[0].copy()
    var L = PLU[1].copy()
    var U = PLU[2].copy()
    print("A == P @ L @ U:", matmul(P, matmul(L, U)))

    var w_v = eigh(A)
    print("eigenvalues:", w_v[0])
```

See the **[Manual»](https://github.com/mojomath/SciJo/blob/main/docs/MANUAL.md)** for a full prose tour of every module, and
**[examples/»](https://github.com/mojomath/SciJo/tree/main/examples)** for runnable, longer versions of the snippets above
(run them all with `examples/run_all.sh`, after `pixi run package`).

## Roadmap

### Near Term
- An analytic-gradient overload for `scijo.optimize.minimize`'s BFGS/CG methods (finite differences only today)
- The general (non-symmetric) eigenvalue problem for `scijo.linalg.eig`/`eigvals`, where eigenvalues may be complex

### Future
- **Optimization**: Curve fitting
- **Signal Processing**: Filtering, windowing, convolution
- **Sparse Matrices**: Efficient storage and operations

## Contributing

Contributions are most welcome! Feel free to add a functionality and open a PR!

Priority areas:
- Algorithm implementations (see Roadmap)
- Performance benchmarks and optimization
- Tests and documentation
- Bug reports and feature requests

## License

Distributed under the Apache 2.0 License with LLVM Exceptions. See the
[license](https://github.com/mojomath/SciJo/blob/main/LICENSE) for more information.

## Acknowledgements

SciJo's API is directly inspired by [SciPy](https://scipy.org), and its test
suite validates numerical results against SciPy and NumPy throughout. Many
thanks to the SciPy and NumPy maintainers and communities for decades of
work on the algorithms, APIs, and documentation that SciJo draws on.

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
