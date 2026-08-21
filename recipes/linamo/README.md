# Linamo <!-- omit in toc -->

![icon](image.jpeg)

Linear algebra for Mojo, specialized for two-dimensional matrices.

[![Version](https://img.shields.io/badge/version-v0.1.0-blue)](https://github.com/mojomath/linamo/releases/tag/v0.1.0)
[![Mojo](https://img.shields.io/badge/mojo-1.0.0-orange)](https://docs.modular.com/mojo/manual/)
[![pixi](https://img.shields.io/badge/pixi%20add-linamo-purple)](https://prefix.dev/channels/modular-community/packages/linamo)
[![CI](https://img.shields.io/github/actions/workflow/status/mojomath/linamo/run_tests.yaml?branch=main&label=tests)](https://github.com/mojomath/linamo/actions/workflows/run_tests.yaml)

**[Manual»](docs/MANUAL.md)** | **[Repository»](https://github.com/mojomath/linamo)** | **[Discord»](https://discord.gg/3rGH87uZTk)**

## Overview

Linamo focuses on efficient **matrix operations** and provides the foundations
for **linear algebra** workflows in Mojo.

| Type           | Information                        |
| -------------- | ---------------------------------- |
| `Matrix`       | A 2-dimensional matrix type        |
| `MatrixView`   | A non-owning view of `Matrix`      |
| `StaticMatrix` | A matrix with a compile-time shape |

The name **Linamo** is **LIN**ear + **A**lgebra + **MO**jo: the field it
covers, and the language it is written in. It can also be read as
**lin**-**amo**: *amo* is Latin for "I love", so the name reads as "I love
linear algebra".

Compared to a general-purpose multi-dimensional array library, Linamo is more
specialized and optimized for linear algebra of 2D matrices. This allows us to
keep the API small, clean, and focused, while still providing powerful
functionality for matrix computations. It is designed to be similar to
`scipy.linalg` in Python and `nalgebra` in Rust, but with a more Mojo-idiomatic
API.

If you need multi-dimensional arrays, consider the
[NuMojo package](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo).

Below are some differences between **Linamo** (this package) and **NuMojo** (a
general-purpose multi-dimensional array library):

| Feature                  | **Linamo**                                | **NuMojo**                                         |
| ------------------------ | ----------------------------------------- | -------------------------------------------------- |
| **Primary goal**         | Linear algebra & matrix computation       | General-purpose ndarray / tensor computing         |
| **Supported dimensions** | 2D only (matrices)                        | Arbitrary dimensions (N-D arrays)                  |
| **Core abstraction**     | Matrix as a mathematical object           | N-dimensional array container                      |
| **Target domain**        | BLAS / LAPACK style workflows             | NumPy-style scientific computing                   |
| **Storage model**        | Matrix-specific storage (row/col strides) | Generic strided N-D storage                        |
| **Static shapes**        | First-class support (compile-time sizes)  | Not a primary focus                                |
| **View semantics**       | Safe read-only + mutable views            | General slicing & broadcasting                     |
| **Indexing model**       | Strict matrix indexing (row, col)         | N-dimensional indexing                             |
| **Negative indexing**    | Not supported (explicit & safe)           | Typically supported                                |
| **Broadcasting**         | Minimal / linear-algebra oriented         | Full NumPy-style broadcasting                      |
| **Specialized kernels**  | Matmul / decompositions / solvers         | Elementwise & tensor ops                           |
| **Performance focus**    | SIMD & BLAS-style kernels                 | Generic tensor operations                          |
| **API philosophy**       | Mathematical clarity & safety             | Flexibility & generality                           |
| **Typical use cases**    | Solvers, decompositions, linear algebra   | Scientific computing, ML preprocessing, tensor ops |

## Goals

The initial goal is to support [Mojo Miji](https://mojo-lang.com/miji/) practice
content, focus on two-dimensional matrix computing, provide simple and intuitive
syntax, and apply a series of targeted optimizations. Throughout the source
code, detailed comments and explanations are provided, under the tag
`[Mojo Miji]` to help readers understand the design decisions and implementation
details.

- Keep the API small and easy to read while learning Mojo and this package.
- Provide simple and intuitive syntax for matrix creation and operations.
- Use **safe Mojo** features and avoid unsafe code as much as possible. The data
  buffer of `Matrix` is a `List` instead of a `Pointer`.
- Differentiate a matrix and a view on the matrix and prevent unintentional
  modifications to the matrices via views.
- Emphasize contiguous storage for 2D matrices, but also support non-contiguous
  views through strides.
- Optimize core operations like matrix multiplication which makes this package a
  better tool if you want to only use 2D matrices.

Linamo targets **Mojo 1.0.0**. The language is stable, but this package's own
API is still moving quickly, so
**pull requests are not accepted at this time**. If you have any suggestions,
questions, or feedback, please feel free to open an
[issue](https://github.com/mojomath/linamo/issues), start a
[discussion](https://github.com/mojomath/linamo/discussions), or reach out on
our [Discord channel](https://discord.gg/3rGH87uZTk). Thank you for your
understanding!

## Install

Linamo is published to the
[modular-community](https://prefix.dev/channels/modular-community/packages/linamo)
channel. Add the channel to your `pixi.toml` and add the package:

```toml
[workspace]
channels = [
    "https://conda.modular.com/max",
    "https://repo.prefix.dev/modular-community",
    "conda-forge",
]
```

```bash
pixi add linamo
```

That brings in Mojo, MAX and [Decimo](https://github.com/forfudan/decimo) as
dependencies, and `import linamo as la` then works with no import path to set.

> **The `pixi add` route arrives with the v0.1.0 release.** Until that tag is
> published, take the package from source, below.

### From source

Clone the repository and let pixi build the environment:

```bash
git clone https://github.com/mojomath/linamo.git
cd linamo
pixi install
pixi run test
```

A program outside the repository compiles against the source tree with the
source directory on the import path, and Decimo beside it:

```bash
LINAMO=/path/to/linamo
mojo run -I $LINAMO/src -I $LINAMO/temp my_program.mojo
```

Or precompile Linamo once and point at the artifact, which is what a build
that imports it repeatedly should do:

```bash
pixi run pack                                     # writes tests/linamo.mojoc
mojo run -I $LINAMO/tests -I $LINAMO/temp my_program.mojo
```

`temp/` is on both lines because Decimo is not optional: the matrix types name
`decimo.Numeric`, so no part of Linamo compiles without it. `pixi run decimo`
resolves and precompiles it there, and every pixi task in this repository
depends on that step, so inside the checkout it needs no separate command.

## Quick start

The [User Manual](docs/MANUAL.md) is the full tour. What follows is enough to
see the shape of the API.

Run the test suite:

```bash
pixi run test
```

### Create matrices

```mojo
import linamo as la

def main() raises:
    # From nested lists
    var A = la.matrix[Float64](
        [[1.0, 2.0, 3.0],
         [4.0, 5.0, 6.0],
         [7.0, 8.0, 10.0]]
    )
    print(A)

    # Convenience constructors
    var I = la.eye[Float64](3)       # 3×3 identity
    var Z = la.zeros[Float64](2, 4)  # 2×4 zeros
    var O = la.ones[Float64](3, 3)   # 3×3 ones
```

### Arithmetic

```mojo
    # `*` and `@` are the matrix product; `**` is repeated multiplication
    var C = A * A    # same call as A @ A
    var D = A**3     # A @ A @ A, by repeated squaring
    var E = A**-1    # inverts first, so this is inv(A)

    # Element-wise arithmetic: `+` and `-` as operators, the rest as methods
    var B = A + O    # addition
    var H = A.mul(A) # Hadamard product
    var Q = A.div(O) # element-wise quotient

    # Scalar operations
    var scaled = A * 2.0
```

### Arbitrary-precision elements

A matrix is parameterised on an element *type*, so
[Decimo](https://github.com/forfudan/decimo)'s exact numbers go in the brackets
where `Float64` would. The operators and routines are the same names; only the
arithmetic underneath differs.

```mojo
import linamo as la
from linamo import BInt, Decimal

def main() raises:
    var a = la.matrix[BInt]([[1, 2], [3, 4]])
    print(a @ a)                 # matrix multiplication, exact
    print(la.trace(a))

    var prices = la.matrix[Decimal]([[Decimal("0.1"), Decimal("0.2")]])
    print(la.sum(prices))        # 0.3, which binary floating point cannot say

    # Elimination runs over a decimal element too: `lu`, `det`, `solve`, `inv`.
    var m = la.from_string[Decimal]("[[4, 7], [2, 6]]")
    print(la.inv(m))             # 0.6 -0.7 / -0.2 0.4, to the type's precision
```

### Linear algebra

```mojo
    # Transpose & trace
    var At = la.transpose(A)
    var t  = la.trace(A)

    # LU decomposition (PA = LU)
    var lup = la.lu(A)
    var L   = lup[0].copy()
    var U   = lup[1].copy()
    var piv = lup[2].copy()

    # Cholesky (A = LL^T, requires SPD matrix)
    var spd = la.matrix[Float64](
        [[4.0, 12.0, -16.0],
         [12.0, 37.0, -43.0],
         [-16.0, -43.0, 98.0]]
    )
    var Lc = la.cholesky(spd)

    # QR decomposition (A = QR)
    var qr_result = la.qr(A)
    var Q = qr_result[0].copy()
    var R = qr_result[1].copy()
```

## Project structure

```text
linamo
├── pixi.toml
├── src/linamo
│   ├── __init__.mojo            # the public surface: `import linamo as la`
│   ├── prelude.mojo
│   ├── types/
│   │   ├── matrix.mojo          # Dynamic Matrix (row/col-major)
│   │   ├── matrix_view.mojo     # Non-owning view with slicing
│   │   ├── matrix_iter.mojo     # Row and column iterators
│   │   ├── static_matrix.mojo   # Compile-time sized Matrix
│   │   └── errors.mojo          # ValueError, IndexError, etc.
│   ├── routines/
│   │   ├── creation.mojo        # matrix, zeros, ones, full, eye, diag, arange, linspace, *_like, from_string
│   │   ├── math.mojo            # add, sub, mul, div, matmul, scalar ops, min, max, prod
│   │   ├── logic.mojo           # comparisons, isclose, logical_*, all, any
│   │   ├── functional.mojo      # fold, apply_along_axis
│   │   ├── manipulation.mojo    # reshape, resize, flatten, contiguous, broadcast_to, astype
│   │   ├── mutation.mojo        # the only source of mutable views: view_mut, fill, assign, store
│   │   ├── searching.mojo       # argmin, argmax
│   │   ├── sorting.mojo         # sort, argsort, sort_inplace
│   │   ├── statistics.mojo      # sum, cumsum
│   │   ├── random.mojo          # rand, seed
│   │   ├── numpy_interop.mojo   # from_numpy, to_numpy
│   │   └── linalg.mojo          # transpose, trace, lu, cholesky, qr, det, solve, inv, lstsq
│   ├── traits/
│   │   └── matrix_like.mojo     # MatrixLike trait
│   └── utils/
│       ├── element.mojo         # compile-time facts about an element type
│       ├── formatting.mojo      # the shared grid every matrix type prints through
│       ├── test_utils.mojo      # assert_matrices_equal / _close
│       ├── indexing.mojo
│       └── str.mojo
├── docs/                        # MANUAL.md (the full tour), ROADMAP.md
├── examples/                    # runnable, one per public type
├── tools/
│   └── ensure_decimo.sh         # resolves and precompiles the decimo dependency
└── tests/
    ├── test_all.sh
    ├── matrix/                   # Matrix creation, indexing, lifecycle, str
    ├── matrix_view/              # View slicing, view-on-view
    ├── static_matrix/            # StaticMatrix tests
    ├── bignum/                   # Matrices of BigInt, BigDecimal, Decimal128
    └── routines/                 # creation, linalg, math, decompositions
```

## Requirements

- Mojo `>=1.0.0,<1.1.0`
- MAX `>=26.5.0,<26.6` — supplies `parallelize()`, which moved out of the Mojo
  standard library in 1.0.0
- [Decimo](https://github.com/forfudan/decimo) — supplies the `Numeric` trait
  the matrix types are written against. `pixi run decimo` resolves and
  precompiles it; `pixi run test`, `examples` and `pack` depend on that task,
  so it needs no separate step

## License

Apache License 2.0. See [LICENSE](LICENSE).
