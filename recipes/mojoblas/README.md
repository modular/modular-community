# mojoBLAS
<!-- omit from toc -->

<p align="center">
  <img src="mojoblas.png" alt="mojoBLAS logo" width="200"/>
</p>

A high-performance **BLAS (Basic Linear Algebra Subprograms)** implementation written in [Mojo](https://modular.com/mojo).

[![Mojo](https://img.shields.io/badge/mojo-1.0.0-orange)](https://docs.modular.com/mojo/manual/)
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
- Mojo `>=1.0.0,<2`

mojoBLAS offers several installation methods to suit different development needs. Choose the method that best fits your workflow:

### Method 1: Git Installation with pixi-build-mojo (Recommended)

Install mojoBLAS directly from the GitHub repository to access both stable releases and cutting-edge features. This method is perfect for developers who want the latest functionality or need to work with the most recent stable version.

Add the following to your existing `pixi.toml`:

```toml
[workspace]
preview = ["pixi-build"]

[package]
name = "your_project_name"
version = "0.1.0"

[package.build]
backend = {name = "pixi-build-mojo", version = "0.*"}

[package.build.config.pkg]
name = "your_package_name"

[package.host-dependencies]
mojo = "==1.0.0"
max-core = "==26.5.0"

[package.build-dependencies]
mojo = "==1.0.0"
max-core = "==26.5.0"
mojoblas = { git = "https://github.com/shivasankarka/mojoBLAS.git", branch = "main" }

[package.run-dependencies]
mojo = "==1.0.0"
max-core = "==26.5.0"
mojoblas = { git = "https://github.com/shivasankarka/mojoBLAS.git", branch = "main" }

[dependencies]
mojo = ">=1.0.0,<2"
max-core = ">=26.5.0,<27"
mojoblas = { git = "https://github.com/shivasankarka/mojoBLAS.git", branch = "main" }
```

Then run:
```bash
pixi install
```

The package will be automatically available in your Pixi environment, and VSCode LSP will provide intelligent code hints.

### Method 2: Stable Release via Pixi (prefix.dev)

For most users, we recommend installing a stable release through Pixi for guaranteed compatibility and reproducibility. `mojoBLAS` is available in the modular-community `https://repo.prefix.dev/modular-community` package repository.

Add the following to your `pixi.toml` file:

```toml
[workspace]
channels = ["https://conda.modular.com/max", "https://repo.prefix.dev/modular-community", "conda-forge"]

[dependencies]
mojoblas = "==0.2.0"
```

Then run:
```bash
pixi install
```

Or, from the `pixi` CLI, run `pixi add mojoblas` in a project whose `channels` already include `https://repo.prefix.dev/modular-community`.

### Method 3: Build Standalone Package

This method creates a portable `mojoblas.mojoc` file that you can use across multiple projects, perfect for offline development or hermetic builds.

1. Clone the repository:
   ```bash
   git clone https://github.com/shivasankarka/mojoBLAS.git
   cd mojoBLAS
   ```

2. Build the package:
   ```bash
   pixi run package
   ```

3. Copy `mojoblas.mojoc` to your project directory or add its parent directory to your include paths.

### Method 4: Direct Source Integration

For maximum flexibility and the ability to modify mojoBLAS source code during development:

1. Clone the repository to your desired location:
   ```bash
   git clone https://github.com/shivasankarka/mojoBLAS.git
   ```

2. When compiling your code, include the mojoBLAS source path:
   ```bash
   mojo run -I "/path/to/mojoBLAS" your_program.mojo
   ```

3. **VSCode LSP Setup** (for code hints and autocompletion):
   - Open VSCode preferences
   - Navigate to `Mojo › Lsp: Include Dirs`
   - Click `Add Item` and enter the full path to your mojoBLAS directory (e.g., `/Users/YourName/Projects/mojoBLAS`)
   - Restart the Mojo LSP server

After setup, VSCode will provide intelligent code completion and hints for mojoBLAS functions!

## Usage

### Basic example

```mojo
from std.memory.alloc import unsafe_alloc
from mojoblas.level1 import dot, axpy, nrm2

def main():
    var x = unsafe_alloc[Scalar[DType.float32]](3)
    var y = unsafe_alloc[Scalar[DType.float32]](3)

    x[unsafe_offset=0] = 1.0
    x[unsafe_offset=1] = 2.0
    x[unsafe_offset=2] = 3.0
    y[unsafe_offset=0] = 4.0
    y[unsafe_offset=1] = 5.0
    y[unsafe_offset=2] = 6.0

    print(dot(3, x, 1, y, 1))
    axpy(3, Float32(2.0), x, 1, y, 1)
    print(y[unsafe_offset=0], y[unsafe_offset=1], y[unsafe_offset=2])
    print(nrm2(3, x, 1))

    x.unsafe_free()
    y.unsafe_free()
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

- `mojoblas/` - Mojo source for BLAS implementations
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

## Changelog

See [docs/CHANGELOG.md](https://github.com/shivasankarka/mojoBLAS/blob/main/docs/CHANGELOG.md) for release notes.

## Contributing

Contributions are welcome. If you find a bug or performance issue, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See [LICENSE](https://github.com/shivasankarka/mojoBLAS/blob/main/LICENSE) for details.

## Acknowledgments

This project is inspired by the Netlib BLAS reference implementation:

http://www.netlib.org/blas/

Special thanks to the Mojo and BLAS communities for the tools and ideas that made this project possible.
