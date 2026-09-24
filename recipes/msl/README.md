# MSL - Mojo Scientific Library <!-- omit from toc -->

<p align="center">
  <img src="msl.png" alt="MSL logo" width="200"/>
</p>

Mojo Scientific Library (MSL) is a comprehensive collection of scientific computation routines derived from the [GNU Scientific Library (GSL)](https://www.gnu.org/software/gsl/) written in pure Mojo, providing special functions, probability distributions, linear algebra, interpolation, numerical differentiation, integration, root-finding, minimization, and ODE solving. Install it with `pixi add msl`.

[![Version](https://img.shields.io/badge/version-v0.2.0-blue)](https://github.com/mojomath/MSL)
[![Mojo](https://img.shields.io/badge/mojo-1.0.0-orange)](https://docs.modular.com/mojo/manual/)

**[Manual»](https://github.com/mojomath/MSL/blob/main/docs/MANUAL.md)** | **[Changelog»](https://github.com/mojomath/MSL/blob/main/docs/changelog.md)** | **[Developer Guide»](https://github.com/mojomath/MSL/blob/main/docs/developer_guide.md)**

MSL is designed as the **low-level scalar backend** for [SciJo](https://github.com/mojomath/SciJo) - the same relationship GSL has with SciPy.

```
SciJo (high-level, user-friendly)
└── MSL (low-level, GSL-like numerics)
```

---

## Overview

| Module | Contents |
|--------|----------|
| `msl.sf` | Special functions (Airy, Bessel, Gamma, Beta, Erf, Legendre, Digamma, Incomplete Gamma/Beta) |
| `msl.integration` | Gauss-Kronrod quadrature (QNG, QK15–QK61, QAG, QAGS) |
| `msl.deriv` | Numerical differentiation (central, forward, backward) |
| `msl.interpolation` | 1-D interpolation (linear, cubic spline, Akima) |
| `msl.optimizer` | Root-finding (bisection, Brent, Newton, secant) and minimization (Brent, golden section) |
| `msl.ode` | ODE solvers (RK4 fixed-step, RKF45 adaptive) |
| `msl.distributions` | 14 probability distributions with samplers and PDFs |
| `msl.rng` | Mersenne Twister RNG with extensible `RNGAlgorithm` trait |
| `msl.vector` | 1-D strided vector with BLAS-backed operations |
| `msl.matrix` | 2-D dense matrix with BLAS-backed operations |
| `msl.blas` | BLAS wrappers (Level 1/2/3) over [mojoBLAS](https://github.com/shivasankarka/mojoBLAS) |
| `msl.linalg` | Dense linear algebra: LU, Cholesky, QR, symmetric eigensolver (Jacobi) |
| `msl.permutation` | Permutation array operations |
| `msl.poly` | Polynomial evaluation, divided-difference interpolation, quadratic/cubic root solving |
| `msl.statistics` | Descriptive statistics: moments, order statistics, robust scale, weighted, Spearman correlation |
| `msl.core` | Constants, error codes (`MSL_*`), math utilities |

### Motivation

Porting GSL is a huge task, but it's much better than writing crazy bindings lol. I also get exposed to more functions that might be helpful in my research, and it helps me learn the math and how to do numerical scientific computing in general (people have come up with crazy tricks!).

I ported a lot of GSL derivative, integration routines, special functions and solvers like `hermv`, ODE solvers like `rkf45`, etc. for my own research project in Mojo. I also ported many statistics routines for SciJo and StatMojo. So I thought it's better to bundle it all together like the GSL library.

I don't plan to port every single function from GSL - perhaps if I get time in the future, I'll try to port more. I'm porting just enough for our libraries: SciJo, StatMojo, and HEPJo. Check out our org [MojoMath](https://github.com/mojomath) for more details.

---

## Installation

MSL requires [pixi](https://pixi.sh) and Mojo 1.0.0.

### With pixi (recommended)

Add the modular-community (`https://repo.prefix.dev/modular-community`) channel and install:

```bash
pixi add msl
```

Or add it directly to your `pixi.toml`:

```toml
[workspace]
channels = ["https://conda.modular.com/max/", "https://repo.prefix.dev/modular-community", "conda-forge"]

[dependencies]
msl = ">=0.2.0"
```

Then run `pixi install`.

### From Git

Add to your `pixi.toml`:

```toml
[workspace]
channels = ["https://conda.modular.com/max/", "conda-forge"]
preview = ["pixi-build"]

[package]
name = "your_project"
version = "x.y.z"

[package.build]
backend = {name = "pixi-build-mojo", version = "0.*"}

[dependencies]
msl = { git = "https://github.com/mojomath/MSL.git", branch = "main" }
```

Then run `pixi install`.

### Local build

```bash
git clone https://github.com/mojomath/MSL.git
cd MSL
pixi run package   # produces msl.mojopkg
```

---

## Quick start

### Special functions

Every special function returns an `SFSResult` with `.val` (computed value), `.err` (absolute error estimate), and `.errno` (0 = success).

```mojo
from msl.sf import airy_ai, bessel_j0, bessel_Jn, gamma, erf, legendre_Pl

def main() raises:
    # Airy function Ai(0) = 0.355028...
    var ai = airy_ai(0.0)
    print(ai.val, ai.err)

    # Bessel J0(1) = 0.765197...
    print(bessel_j0(1.0).val)

    # Integer-order Bessel J_n(x)
    print(bessel_Jn(5, 2.0).val)   # J5(2)

    # Gamma(5) = 24
    print(gamma(5.0).val)

    # erf(1) = 0.842700...
    print(erf(1.0).val)

    # Legendre P_10(0.5)
    print(legendre_Pl(10, 0.5).val)
```

### Random numbers and distributions

```mojo
from msl.rng import RNG, MT19937
from msl.distributions import gaussian, uniform, gamma, poisson, tdist, weibull

def main():
    var rng = RNG[MT19937](MT19937(42), 42)

    print(rng.uniform())              # uniform in [0,1)
    print(gaussian(rng, 1.0))        # N(0, sigma=1)
    print(gamma(rng, 2.0, 1.0))      # Gamma(shape=2, scale=1)
    print(poisson(rng, 5.0))         # Poisson(lambda=5)
    print(tdist(rng, 10.0))          # Student-t(nu=10)
    print(weibull(rng, 1.0, 2.0))    # Weibull(scale=1, shape=2)
```

### Numerical integration

```mojo
from msl.integration import qags, qag, qng_integrate, MSL_INTEG_GAUSS21

def main() raises:
    # Adaptive integration
    def integrand(x: Float64) capturing -> Float64:
        return x * x

    var r = qags[integrand](0.0, 1.0, 1e-10, 1e-10)
    print(r.val, r.err)   # ≈ 0.333333

    # General adaptive QAG with rule selection
    var r2 = qag[integrand](0.0, 1.0, 1e-10, 1e-10, key=MSL_INTEG_GAUSS21)
    print(r2.val)
```

### Numerical differentiation

```mojo
from msl.deriv import deriv_central, deriv_forward

def main():
    def sin_fn(x: Float64) capturing -> Float64:
        from std.math import sin
        return sin(x)

    # d/dx sin(x) at x=1 ≈ cos(1) = 0.540302...
    var r = deriv_central[sin_fn](1.0)
    print(r.val, r.err)
```

### Interpolation

```mojo
from std.memory.alloc import unsafe_alloc
from msl.interpolation import CubicSpline, AkimaSpline

def main() raises:
    var xa = unsafe_alloc[Float64](5)
    var ya = unsafe_alloc[Float64](5)
    xa[unsafe_offset=0]=0.0; xa[unsafe_offset=1]=1.0; xa[unsafe_offset=2]=2.0
    xa[unsafe_offset=3]=3.0; xa[unsafe_offset=4]=4.0
    ya[unsafe_offset=0]=0.0; ya[unsafe_offset=1]=1.0; ya[unsafe_offset=2]=4.0
    ya[unsafe_offset=3]=9.0; ya[unsafe_offset=4]=16.0

    var spline = CubicSpline(xa, ya, 5)
    print(spline.eval(1.5).val)   # ≈ 2.25 (interpolates x^2)
    print(spline.deriv(2.0).val)  # ≈ 4.0  (derivative of x^2)
    xa.unsafe_free(); ya.unsafe_free()
```

### Root-finding and minimization

```mojo
from msl.optimizer import root_brent, root_newton, min_brent
from std.math import sin, cos

def main():
    # Brent root: sin(x) = 0 near pi
    def fn_(x: Float64) capturing -> Float64:
        return sin(x)

    var r = root_brent[fn_](3.0, 4.0)
    print(r.root, r.nit)   # ≈ 3.14159...

    # Brent minimization: -(cos(x)) in [-1, 1]
    def neg_cos(x: Float64) capturing -> Float64:
        return -cos(x)

    var m = min_brent[neg_cos](-1.0, 0.5, 1.0)
    print(m.x, m.fun)   # ≈ 0.0, -1.0
```

### ODE solving

```mojo
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from msl.ode import ode_rk4, ode_rkf45
from std.math import cos, sin

def main() raises:
    # Harmonic oscillator: y'' + y = 0
    # y0(t) = cos(t),  y1(t) = -sin(t)
    def rhs[o1: MutOrigin, o2: MutOrigin, //](
        t: Float64,
        y: Pointer[Float64, o1],
        dydt: Pointer[Float64, o2],
    ) capturing:
        dydt[unsafe_offset=0] = y[unsafe_offset=1]
        dydt[unsafe_offset=1] = -y[unsafe_offset=0]

    var y = unsafe_alloc[Float64](2)
    y[unsafe_offset=0] = 1.0; y[unsafe_offset=1] = 0.0

    # Fixed-step RK4
    _ = ode_rk4[rhs](0.0, 1.0, 0.01, y, 2)
    print(y[unsafe_offset=0], cos(1.0))   # ≈ cos(1)

    # Adaptive RKF45
    y[unsafe_offset=0] = 1.0; y[unsafe_offset=1] = 0.0
    var r2 = ode_rkf45[rhs](0.0, 1.0, 0.1, y, 2, epsabs=1e-8, epsrel=1e-8)
    print(r2.nsteps, "steps")
    y.unsafe_free()
```

### Vectors and matrices (with BLAS backend)

```mojo
from std.memory.alloc import unsafe_alloc
from msl.vector import Vector
from msl.matrix import Matrix
from msl.blas import blas_dot, blas_gemm

def main() raises:
    # Owning vectors
    var x = Vector(3, initialize=True)
    var y = Vector(3, initialize=True)
    x[0]=1.0; x[1]=2.0; x[2]=3.0
    y[0]=4.0; y[1]=5.0; y[2]=6.0

    print(blas_dot(x, y))   # 32.0

    # Non-owning view over external buffer (zero-copy, e.g. from NuMojo NDArray)
    var buf = unsafe_alloc[Float64](6)
    # ... fill buf from NDArray.unsafe_ptr() ...
    _ = Vector(buf, 6)
    # Vector's owner field is False here: destructor does NOT free buf
    buf.unsafe_free()

    # Matrix-matrix multiply via mojoBLAS
    var A = Matrix(2, 3, initialize=True)
    var B = Matrix(3, 2, initialize=True)
    var C = Matrix(2, 2, initialize=True)
    # ... fill A and B ...
    blas_gemm(A, B, C)   # C = A @ B
```

---

## API reference

### Special functions (`msl.sf`)

| Function | Description |
|----------|-------------|
| `airy_ai`, `airy_bi` | Airy functions Ai, Bi and scaled/derivative variants |
| `bessel_j0`, `bessel_j1`, `bessel_Jn` | Bessel J (cylindrical, first kind) |
| `bessel_y0`, `bessel_y1`, `bessel_Yn` | Bessel Y (cylindrical, second kind) |
| `bessel_i0_scaled`, `bessel_i1_scaled`, `bessel_In` | Modified Bessel I |
| `bessel_k0_scaled`, `bessel_k1_scaled`, `bessel_Kn` | Modified Bessel K |
| `gamma`, `lngamma`, `gammastar`, `gammainv` | Gamma function and variants |
| `factorial`, `double_factorial`, `ln_factorial` | Factorial functions |
| `beta`, `lnbeta` | Beta function |
| `erf`, `erfc`, `log_erfc`, `erf_Z`, `erf_Q`, `hazard` | Error functions |
| `legendre_P1`, `legendre_P2`, `legendre_P3`, `legendre_Pl` | Legendre polynomials |
| `psi`, `psi_n` | Digamma and polygamma |
| `gamma_inc`, `gamma_inc_P`, `gamma_inc_Q` | Incomplete gamma functions |
| `beta_inc` | Regularized incomplete beta I_x(a,b) |

### Integration (`msl.integration`)

| Function | Description |
|----------|-------------|
| `qng_integrate` | Non-adaptive 10→21→43→87-point Gauss-Kronrod |
| `qk15` .. `qk61` | Fixed-order Gauss-Kronrod rules |
| `qag` | General adaptive integration (choice of 6 rules) |
| `qags` | Adaptive with Wynn epsilon extrapolation |

### Distributions (`msl.distributions`)

Gaussian, uniform, exponential, gamma, beta, chi-squared, Poisson, Student-t, log-normal, Weibull, binomial, negative binomial, Cauchy, Laplace. Each has a sampler and a PDF function.

### ODE solvers (`msl.ode`)

| Function | Description |
|----------|-------------|
| `ode_rk4` | Classical 4th-order Runge-Kutta, fixed step |
| `ode_rkf45` | Runge-Kutta-Fehlberg 4(5), adaptive step with error control |

### Optimizer (`msl.optimizer`)

| Function | Description |
|----------|-------------|
| `root_bisect` | Bisection (bracketing, guaranteed convergence) |
| `root_brent` | Brent's method (bracketing + interpolation) |
| `root_newton` | Newton-Raphson (requires derivative) |
| `root_secant` | Secant method (derivative-free) |
| `min_brent` | Brent minimization (parabolic interpolation + golden section) |
| `min_golden` | Golden section search |

### RNG (`msl.rng`)

```mojo
# Use the default MT19937 algorithm
var rng = RNG[MT19937](MT19937(seed), seed)

# Implement your own algorithm
struct MyRNG(RNGAlgorithm):
    def next(mut self) -> UInt64: ...
    def seed(mut self, s: UInt64): ...
```

### Linear algebra (`msl.linalg`)

| Function | Description |
|----------|-------------|
| `lu_decomp`, `lu_svx`, `lu_solve` | LU decomposition with partial pivoting and solve |
| `lu_det`, `lu_lndet` | Determinant and log-determinant from an LU factorization |
| `lu_invert` | Matrix inverse from an LU factorization |
| `cholesky_decomp`, `cholesky_svx`, `cholesky_solve` | Cholesky decomposition and solve (SPD matrices) |
| `qr_decomp`, `qr_svx`, `qr_solve` | QR decomposition (Householder) and solve |
| `eigen_jacobi` | Symmetric eigenvalue/eigenvector solver (cyclic Jacobi) |

### Polynomials (`msl.poly`)

| Function | Description |
|----------|-------------|
| `poly_eval`, `poly_eval_derivs` | Horner-form polynomial evaluation and derivatives |
| `poly_dd_init`, `poly_dd_eval`, `poly_dd_taylor` | Newton divided-difference interpolation |
| `poly_solve_quadratic`, `poly_solve_cubic` | Real/complex roots of quadratics and cubics |

### Statistics (`msl.statistics`)

| Function group | Description |
|----------------|-------------|
| `stats_mean`, `stats_variance`, `stats_sd`, `stats_skew`, `stats_kurtosis` | Central moments |
| `stats_covariance`, `stats_correlation`, `stats_lag1_autocorrelation` | Association and autocorrelation |
| `stats_median`, `stats_quantile_from_sorted_data`, `stats_trmean_from_sorted_data` | Order statistics (require pre-sorted data) |
| `stats_mad`, `stats_Sn_from_sorted_data`, `stats_Qn_from_sorted_data` | Robust scale estimators (Rousseeuw & Croux) |
| `stats_wmean`, `stats_wvariance`, `stats_wskew`, `stats_wkurtosis` | Weighted counterparts of the above |
| `stats_ttest`, `stats_pvariance` | Two-sample t-test and pooled variance |
| `stats_spearman` | Spearman rank correlation |

---

## Design

MSL follows the GSL design philosophy:

- **Scalar-first**: all functions operate on `Float64` scalars; array operations belong in SciJo.
- **Error propagation**: every special function returns `SFSResult` with `.val`, `.err`, `.errno`
- **Zero ownership surprises**: `Vector` and `Matrix` support non-owning views via `Vector(ptr, n)` / `Matrix(ptr, rows, cols)` for zero-copy interop with NuMojo NDArray! 
- **Extensible RNG**: the `RNGAlgorithm` trait lets you plug in any generator; distributions are parametric over `T: RNGAlgorithm`
- **BLAS-backed linalg**: `msl.blas` wraps mojoBLAS - same pattern as GSL wrapping CBLAS

---

## Tests

```bash
pixi run tests
```

Runs the full test suite (330 tests covering all modules).

---

## Status

| Module | Status |
|--------|--------|
| `msl.sf` | ✅ Airy, Bessel (J/Y/I/K orders 0/1/n), Gamma, Beta, Erf, Legendre, Digamma, Incomplete Gamma/Beta |
| `msl.integration` | ✅ QNG (87-pt), QK15–QK61, QAG, QAGS |
| `msl.deriv` | ✅ Central (5-pt), forward/backward (4-pt), adaptive step refinement |
| `msl.interpolation` | ✅ Linear, natural cubic spline, Akima |
| `msl.optimizer` | ✅ Bisect, Brent root, Newton, secant, Brent min, golden section |
| `msl.ode` | ✅ RK4, RKF45 with adaptive step control |
| `msl.distributions` | ✅ 14 distributions with samplers and PDFs |
| `msl.rng` | ✅ MT19937, `RNGAlgorithm` trait |
| `msl.vector` | ✅ Strided vector, math ops, BLAS-backed |
| `msl.matrix` | ✅ Dense matrix, math ops, BLAS-backed |
| `msl.blas` | ✅ Level 1/2/3 wrappers over mojoBLAS |
| `msl.linalg` | ✅ LU, Cholesky, QR, symmetric eigensolver (Jacobi) |
| `msl.permutation` | ✅ Permutation array |
| `msl.poly` | ✅ Evaluation, divided-difference interpolation, quadratic/cubic solving |
| `msl.statistics` | ✅ Moments, order statistics, robust scale (MAD/Sn/Qn), weighted, Spearman |

---

## Roadmap

- [ ] FFT (pure Mojo Cooley-Tukey from SciJo)
- [ ] Complete Bessel spherical functions (j_n, y_n)
- [ ] Elliptic integrals (K, E, Π)
- [ ] Hypergeometric functions (2F1, 1F1)
- [ ] Implicit ODE solvers (RK4-implicit, BDF)
- [ ] Sparse matrices
- [x] Wiring it to SciJo (`scijo.integrate` and `scijo.optimize` already call into MSL).

---

## Contributing

Bug reports and PRs are welcome. This library is part of the [MojoMath](https://github.com/mojomath) organisation alongside SciJo, MatMojo and StatMojo.

---

## License

GPL-3.0-or-later. See [LICENSE](LICENSE) for details.

Derived from the GNU Scientific Library (GSL). Original GSL code is Copyright (C) 1996–2007 Gerard Jungman, Brian Gough and contributors. The BLAS layer uses [mojoBLAS](https://github.com/shivasankarka/mojoBLAS) (MIT).
