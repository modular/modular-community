# mojoBLAS
<!-- omit from toc -->

<p align="center">
  <img src="mojoblas.png" alt="mojoBLAS logo" width="200"/>
</p>

A high-performance **BLAS (Basic Linear Algebra Subprograms)** implementation written in [Mojo](https://modular.com/mojo).

[![Mojo](https://img.shields.io/badge/mojo-1.0.0b1-orange)](https://docs.modular.com/mojo/manual/)
[![Tests](https://img.shields.io/badge/tests-level1%2F2%2F3-brightgreen)]()
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

## Overview

`mojoBLAS` is a pure-Mojo BLAS implementation focused on performance. It currently includes:

- **Level 1 BLAS**: vector-vector operations such as `dot`, `axpy`, `nrm2`, `scal`, and more.
- **Level 2 BLAS**: matrix-vector operations such as `gemv`, `ger`, triangular and packed matrix-vector routines.
- **Level 3 BLAS**: matrix-matrix operations such as `gemm`, `syrk`, `syr2k`, `symm`, `trmm`, and `trsm`.
- **Benchmarking suite**: comparison against reference/system BLAS implementations.

The codebase is currently optimized for real scalar data types through Mojo `DType` support.

## Installation

### Prerequisites

- Pixi
- Mojo `>=1.0.0b1,<2`

### Modular community
`mojoBLAS` is available in the modular-community `https://repo.prefix.dev/modular-community` package repository. Add the following to your `channels` list in your `pixi.toml` file:

```toml
channels = ["https://conda.modular.com/max", "https://repo.prefix.dev/modular-community", "conda-forge"]
```

Then, you can install `mojoBLAS` using any of these methods:

1. From the `pixi` CLI, run the command ```pixi add mojoblas```.

2. In the `pixi.toml` file of your project, add the following dependency:
    ```toml
    mojoblas = "==0.1.0"
    ```
Then run `pixi install` to download and install the package.

### Use as a dependency

Add the repository to your `pixi.toml`:

```toml
[workspace]
preview = ["pixi-build"]

[dependencies]
mojo = ">=1.0.0b1,<2"
mojoblas = { git = "https://github.com/shivasankarka/mojoBLAS.git", branch = "main" }
```

Then run:

```bash
pixi install
```

### Clone locally

```bash
git clone https://github.com/shivasankarka/mojoBLAS.git
cd mojoBLAS
pixi install
```

## Usage

### Basic example

```mojo
from mojoblas.src.level1 import dot, axpy, nrm2

fn main():
    var x = alloc[Float32](3)
    var y = alloc[Float32](3)

    x[0] = 1.0
    x[1] = 2.0
    x[2] = 3.0
    y[0] = 4.0
    y[1] = 5.0
    y[2] = 6.0

    print(dot(3, x, 1, y, 1))
    axpy(3, 2.0, x, 1, y, 1)
    print(y[0], y[1], y[2])
    print(nrm2(3, x, 1))

    x.free()
    y.free()
```

### Available routines

- **Level 1**: `asum`, `axpy`, `copy`, `dot`, `iamax`, `nrm2`, `rot`, `rotg`, `rotm`, `rotmg`, `scal`, `swap`
- **Level 2**: `gbmv`, `gemv`, `ger`, `sbmv`, `spmv`, `spr`, `spr2`, `symv`, `syr`, `syr2`, `tbmv`, `tbsv`, `tpmv`, `tpsv`, `trmv`, `trsv`
- **Level 3**: `gemm`, `symm`, `syrk`, `syr2k`, `trmm`, `trsm`

## Testing

Run the test suites with Pixi:

```bash
pixi run test_level1
pixi run test_level2
pixi run test_level3
```

## Benchmarking

The repository includes benchmark scripts. This benchmark compares mojoblas against general openblas and Accelerate (on Apple M chips) routines. To run the full benchmarks and generate plots, run the following command

```bash
pixi run -e bench bench_all
```

### Outputs

- `benchmarks/bench_plot_level1.png`
- `benchmarks/bench_plot_level2.png`
- `benchmarks/bench_plot_level3.png`

## Project structure

- `src/` - Mojo source for BLAS implementations
- `tests/` - Mojo tests and reference data
- `benchmarks/` - benchmark scripts and plots
- `docs/` - Reference documentation.

## Roadmap

### Completed

- [x] Level 1 BLAS
- [x] Level 2 BLAS
- [x] Level 3 BLAS
- [x] Benchmarking suite

### Future goals

- [ ] Optimize current algorithms (Goal: openblas, accelerate performance and more :))
- [ ] Complex number support
- [ ] GPU acceleration

## Contributing

Contributions are welcome. If you find a bug or performance issue, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

This project is inspired by the Netlib BLAS reference implementation:

http://www.netlib.org/blas/

Special thanks to the Mojo and BLAS communities for the tools and ideas that made this project possible.
