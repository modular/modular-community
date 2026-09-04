# hdf5-mojo

High-level HDF5 bindings for Mojo: Read and write HDF5 files using an h5py-compatible API that wraps the HDF5 C library.

## Overview

`hdf5-mojo` wraps the HDF5 C library and exposes it through a high-level, h5py-style API: open/create files, create groups and datasets, read and write array data, and manage attributes, all with Mojo-native types.

It covers most of the basic features needed for working with datasets. Full HDF5 feature parity is not a goal yet — see [Known limitations](#known-limitations) for what's missing.

## Motivation

I'm working on porting some particle physics simulation libraries to Mojo. Since HDF5 is widely used there, I wrote these bindings to make it easier to use HDF5 datasets directly from Mojo, and built out the features my own projects needed first. Full HDF5 feature parity might come later if I get more free time.

## Features

- Read/write HDF5 files with h5py-style API
- Create groups, datasets, and attributes
- Support for `float64`, `float32`, `int32`, `int64`, `int8`, `uint8` dtypes
- N-D Array reading with `read[dtype]()` (uses NuMojo NDArray in the backend.)
- Chunked datasets, resizable (`maxshape`) datasets, and fill values
- Built-in filters: gzip compression, shuffle, and fletcher32 checksums,
  plus filter metadata accessors (`compression()`, `shuffle()`, etc.)
- Hyperslab, slice (`read_slice`/`write_slice`), and point
  (`read_point`/`write_point`) I/O on datasets
- Scalar dataset support, including h5py-style `dset[()]` indexing
- `require_group` / `require_dataset` helpers, `get_opt()` for non-raising
  lookups
- Automatic library discovery via pixi

## Supported DType mappings

| Mojo `DType` | HDF5 C type |
|---|---|
| `DType.float64` | `H5T_NATIVE_DOUBLE` |
| `DType.float32` | `H5T_NATIVE_FLOAT` |
| `DType.int32` | `H5T_NATIVE_INT32` |
| `DType.int64` | `H5T_NATIVE_INT64` |
| `DType.int8` | `H5T_NATIVE_INT8` |
| `DType.uint8` | `H5T_NATIVE_UINT8` |

## Known limitations

This library covers the dataset/group/attribute workflows needed for the
particle-physics use case it was built for; it is not yet full h5py parity.
Notably missing, as of v0.2:

- **Types**: no `int16`, `uint16`, `uint32`, `uint64`, `bool`, or string
  (fixed- or variable-length) datasets/attributes. No compound, enum,
  opaque, or reference types.
- **Indexing**: no fancy indexing or multi-block hyperslab selections;
  slicing is limited to simple rectangular regions with unit stride.
- **Links & traversal**: no `visit()`/`visititems()`, no hard/soft/external
  links, no `move`/`copy`.
- **File-level features**: no `libver` bounds, alternate file drivers
  (core/in-memory, family/split), or chunk cache tuning.
- **Advanced HDF5**: no dimension scales, virtual datasets, SWMR, parallel
  HDF5/MPI, or object/region references.

If one of these blocks you, please open an issue — it helps prioritize the
roadmap.

## Installation

Add the following to your `pixi.toml`:

```toml
[workspace]
preview = ["pixi-build"]

[package]
name = "your_project_name"
version = "x.y.z"

[package.build]
backend = {name = "pixi-build-mojo", version = "0.*"}

[package.build.config.pkg]
name = "your_package_name"

[package.host-dependencies]
mojo = "==1.0.0"

[package.build-dependencies]
mojo = "==1.0.0"
hdf5-mojo = { git = "https://github.com/shivasankarka/hdf5-mojo.git", branch = "main"}

[package.run-dependencies]
mojo = "==1.0.0"
hdf5-mojo = { git = "https://github.com/shivasankarka/hdf5-mojo.git", branch = "main"}

[dependencies]
mojo = ">=1.0.0,<2"
hdf5-mojo = { git = "https://github.com/shivasankarka/hdf5-mojo.git", branch = "main"}
```

Then run:

```bash
pixi install
```

## Quickstart

### Reading a file

```mojo
from hdf5 import File, f64, i32

def main() raises:
    var f = File("data.h5", "r")

    # Access datasets and groups with .get() method
    var obj = f.get("group/dataset")
    if obj.is_dataset():
        var dset = obj.dataset()
        print(dset.shape())   # [100, 50]
        print(dset.dtype())   # "float64"
        
        # Read all data into an NuMojo NDArray
        var arr = dset.read[f64]()
        print("arr: ", arr)

    # Iterate over root-level items
    for name in f.keys():
        print(name)

    # Read attributes
    var version = f.attrs().get[i32]("version", Int32(0))

    f.close()
```

### Writing a file

```mojo
from hdf5 import File, f64, i32

def main() raises:
    var f = File("output.h5", "w")

    # Create groups (nested paths are automatically created)
    f.create_group("results/nested")

    # Create datasets
    var shape = List[Int](100, 50)
    var dset = f.create_dataset[f64]("results/data", shape)

    # Write data
    var n = 100 * 50
    var buf = alloc[Scalar[f64]](n)
    # ... fill buf with data ...
    dset.write[DType.float64](buf, n)
    buf.free()
    # You can also pass a NDArray directly as buffer.

    # Write attributes
    f.attrs().set[i32]("version", Int32(1))

    f.close()
```

## API Reference

See [docs/api_reference.md](docs/api_reference.md) for the complete API documentation.

---

## Examples

See `examples/demo_api.mojo` for a complete example demonstrating the h5py-style API with an 
example dataset. 

## Project structure

```
hdf5/
  __init__.mojo       # Package entry point
  ffi.mojo            # Low-level HDF5 C FFI wrapper (HDF5Lib)
  core.mojo           # High-level h5py-compatible API
examples/
  demo_api.mojo       # Demo file.
  demo_data.h5        # Sample HDF5 file.
```

## Troubleshooting

**File fails to open** — Ensure `libhdf5.dylib` is accessible. If using pixi, run `pixi install` first so `$CONDA_PREFIX` is set.

**Memory leak warnings** — Call `.close()` on `File` when done.

**"dataset already exists" on write** — The write helpers refuse to overwrite existing datasets. Choose a new path or recreate the file.

## License

Distributed under the Apache 2.0 License. See `LICENSE` for details.

## Acknowledgement

Huge thanks to the HDF5 maintainers, this cool library exists thanks to their work :)

## Contributing

Contributions are always welcome! If you think there's a feature missing, make an issue or give it a try and make a PR! 

## Contact

For questions or bug reports, please open an issue in the repository.
