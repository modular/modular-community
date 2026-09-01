# Linamo <!-- omit in toc -->

![icon](image.jpeg)

Linear algebra for Mojo, specialized for two-dimensional matrices.

## Overview

Linamo focuses on efficient **matrix operations** and provides the foundations
for **linear algebra** workflows in Mojo. It offers three matrix types --- an
owning `Matrix`, a non-owning `MatrixView`, and a compile-time shaped
`StaticMatrix` --- together with creation, manipulation, and decomposition
routines (`lu`, `cholesky`, `qr`, `det`, `solve`, `inv`, `lstsq`, ...).

Each type is parameterised on an element **type** rather than a `DType`, so the
same operators and routines run over fixed-width numbers (`Float64`, `Int32`,
... through SIMD kernels) and over exact ones from
[Decimo](https://github.com/forfudan/decimo) --- arbitrary-precision `BInt`,
and base-ten `Decimal`, where `0.1 + 0.2` is `0.3`. `la.matrix[Float64]` and
`la.matrix[BInt]` differ only in the brackets.

Compared to a general-purpose multi-dimensional array library, Linamo is more
specialized and optimized for linear algebra of 2D matrices. This keeps the API
small, clean, and focused, in the spirit of `scipy.linalg` in Python and
`nalgebra` in Rust, but with a more Mojo-idiomatic API. If you need
multi-dimensional arrays, consider the
[NuMojo package](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo).

The name **Linamo** is **LIN**ear + **A**lgebra + **MO**jo: the field it
covers, and the language it is written in. It can also be read as
**lin**-**amo**: *amo* is Latin for "I love", so the name reads as "I love
linear algebra".

## Documentation

The [README](https://github.com/mojomath/linamo#readme) has a quick start and
build-from-source instructions; the
[User Manual](https://github.com/mojomath/linamo/blob/main/docs/MANUAL.md) is
the full tour.

## License

Apache License 2.0. See
[LICENSE](https://github.com/mojomath/linamo/blob/main/LICENSE).
