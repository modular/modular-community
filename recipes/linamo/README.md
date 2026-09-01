# Linamo <!-- omit in toc -->

![icon](image.jpeg)

Linear algebra for Mojo, specialized for two-dimensional matrices.

[![Version](https://img.shields.io/badge/version-v0.1.0-blue)](https://github.com/mojomath/linamo/releases/tag/v0.1.0)
[![Mojo](https://img.shields.io/badge/mojo-1.0.0-orange)](https://docs.modular.com/mojo/manual/)
[![pixi](https://img.shields.io/badge/pixi%20add-linamo-purple)](https://prefix.dev/channels/modular-community/packages/linamo)
[![CI](https://img.shields.io/github/actions/workflow/status/mojomath/linamo/run_tests.yaml?branch=main&label=tests)](https://github.com/mojomath/linamo/actions/workflows/run_tests.yaml)

**[Manual»](docs/MANUAL.md)** | **[Changelog»](docs/CHANGELOG.md)** | **[Repository»](https://github.com/mojomath/linamo)** | **[Discord»](https://discord.gg/3rGH87uZTk)**

## Overview

Linamo focuses on efficient **matrix operations** and provides the foundations
for **linear algebra** workflows in Mojo.

| Type           | Information                        |
| -------------- | ---------------------------------- |
| `Matrix`       | A 2-dimensional matrix type        |
| `MatrixView`   | A non-owning view of `Matrix`      |
| `StaticMatrix` | A matrix with a compile-time shape |

Each is parameterised on an element **type** rather than a `DType`, so the same
operators and routines run over fixed-width numbers and over exact ones:

| Element type            | Information                                            |
| ----------------------- | ------------------------------------------------------ |
| `Float64`, `Int32`, ... | Any Mojo scalar, through SIMD kernels                  |
| `BInt`                  | Arbitrary-precision integer, with no width to overflow |
| `Decimal`, `Dec128`     | Base-ten decimals, so `0.1 + 0.2` is `0.3`             |

`la.matrix[Float64]` and `la.matrix[BInt]` differ only in the brackets. The
exact types come from [Decimo](https://github.com/forfudan/decimo) and are
re-exported, so reaching for one costs no second import -- see
[Arbitrary-precision elements](#arbitrary-precision-elements).

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

| Feature                  | **Linamo**                                                   | **NuMojo**                                         |
| ------------------------ | ------------------------------------------------------------ | -------------------------------------------------- |
| **Primary goal**         | Linear algebra & matrix computation                          | General-purpose ndarray / tensor computing         |
| **Supported dimensions** | 2D only (matrices)                                           | Arbitrary dimensions (N-D arrays)                  |
| **Core abstraction**     | Matrix as a mathematical object                              | N-dimensional array container                      |
| **Element types**        | Any Mojo scalar, plus arbitrary-precision `BInt` / `Decimal` | `DType` scalars (`NDArray[dtype: DType]`)          |
| **Target domain**        | BLAS / LAPACK style workflows                                | NumPy-style scientific computing                   |
| **Storage model**        | Matrix-specific storage (row/col strides)                    | Generic strided N-D storage                        |
| **Static shapes**        | First-class support (compile-time sizes)                     | Not a primary focus                                |
| **View semantics**       | Safe read-only + mutable views                               | General slicing & broadcasting                     |
| **Indexing model**       | Strict matrix indexing (row, col)                            | N-dimensional indexing                             |
| **Negative indexing**    | Not supported (explicit & safe)                              | Typically supported                                |
| **Broadcasting**         | Minimal / linear-algebra oriented                            | Full NumPy-style broadcasting                      |
| **Specialized kernels**  | Matmul / decompositions / solvers                            | Elementwise & tensor ops                           |
| **Performance focus**    | SIMD & BLAS-style kernels                                    | Generic tensor operations                          |
| **API philosophy**       | Mathematical clarity & safety                                | Flexibility & generality                           |
| **Typical use cases**    | Solvers, decompositions, linear algebra                      | Scientific computing, ML preprocessing, tensor ops |

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
source directory on the import path, and `temp/` beside it:

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

Decimo is not optional --- the matrix types name `decimo.Numeric`, so no part
of Linamo compiles without it --- but it is an ordinary workspace dependency,
so `pixi install` puts it on the import path and `temp/` is normally empty.
`pixi run decimo` is what confirms that, and it is also the escape hatch: set
`DECIMO_PATH` to a local checkout, or `LINAMO_DECIMO=git` to build the pinned
upstream commit, and it precompiles Decimo into `temp/`, which is why `-I temp`
is on both lines above. Every pixi task in this repository depends on that
step, so inside the checkout it needs no separate command.

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
where `Float64` would. The operators and routines keep their names; only the
arithmetic underneath differs. What changes is the range: `BInt` has no width
to overflow, and `Decimal` is base ten, so a decimal fraction is stored as
written.

```mojo
import linamo as la          # the element types come with it: la.BInt, la.Decimal

def main() raises:
    # The Fibonacci matrix to the 100th power carries F(101) in its corner:
    # 21 digits, where `Int64` runs out at 19. `**` climbs by repeated
    # squaring, so this is nine matrix products rather than a hundred.
    var fib = la.matrix[la.BInt]([[1, 1], [1, 0]])
    print((fib**100)[0, 0])      # 573147844013817084101

    # A product outgrows the entries it came from, and every digit is kept.
    var big = la.from_string[la.BInt](
        "[[123456789012345678901234567890, 2],"
        + " [3, 987654321098765432109876543210]]"
    )
    print((big @ big)[0, 0])     # all 59 digits of it, exactly:
    # 15241578753238836750495351562536198787501905199875019052106

    # Decimals add without drift. In `Float64` this same sum comes to
    # 0.9999999999999999.
    print(la.sum(la.full[la.Decimal](1, 10, la.Decimal("0.1"))))  # 1.0
```

The gain is not only tidier output. `0.1 * 0.6 - 0.2 * 0.3` is exactly zero,
so the matrix below is singular --- but in binary floating point the
determinant misses zero by one rounding step, `inv` believes it, and what
comes back is not an inverse of anything:

```mojo
    var f = la.matrix[Float64]([[0.1, 0.2], [0.3, 0.6]])
    print(la.det(f))             # 1.1102230246251562e-18, not 0.0
    print(la.inv(f))             # entries around 5e17 --- nonsense, unflagged

    # The same elimination over `Decimal` reaches zero exactly and stops.
    var d = la.from_string[la.Decimal]("[[0.1, 0.2], [0.3, 0.6]]")
    print(la.inv(d))             # ValueError: Coefficient matrix A is singular.
```

`Decimal` is the element type to reach for there. The routines that run
elimination --- `lu`, `det`, `solve`, `inv`, `lstsq` --- divide, so they mean
whatever `/` means on the element type: over `Decimal` that is a quotient at
the type's precision, but over `BInt` it truncates, and elimination then
returns whole numbers that are not the answer without raising. The
[manual](docs/MANUAL.md#arbitrary-precision-elements) has the details.

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
│   ├── errors.mojo              # the error kinds, re-exported from decimo
│   ├── types/
│   │   ├── matrix.mojo          # Dynamic Matrix (row/col-major)
│   │   ├── matrix_view.mojo     # Non-owning view with slicing
│   │   ├── matrix_iter.mojo     # Row and column iterators
│   │   └── static_matrix.mojo   # Compile-time sized Matrix
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
├── docs/                        # MANUAL.md (the full tour), CHANGELOG.md, ROADMAP.md
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
- [Decimo](https://github.com/forfudan/decimo) `>=0.13.0,<0.14` — supplies the
  `Numeric` and `Parsable` traits the matrix types are written against, and the
  error kinds in `linamo.errors`. It is a workspace dependency, so `pixi
  install` brings it in; `pixi run decimo` resolves it and is the hook for
  building against a local or unreleased Decimo instead

## License

Apache License 2.0. See [LICENSE](LICENSE).
