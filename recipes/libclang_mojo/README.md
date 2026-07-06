# mojo_libclang

High-level Mojo bindings for LLVM `libclang`.

`mojo_libclang` provides a Python `clang.cindex`-style API for source-code tooling in Mojo.

## Highlights

- Exposes most of `libclang` API for `C`, `C++`, and some `obj-C`
- Parse C and C++ source files with libclang.
- Inspect cursor spelling, kind, location, extent, parents, and type.
- Read diagnostics with formatted messages.
- Query canonical types, pointee types, result types, fields, and declarations.

## Example

```c
//math_api.h
typedef struct Point { int x; int y; } Point;
int add(int a, int b);
```

```mojo
# main.mojo
from clang.cindex import CursorKind, Index


def main() raises:
    var index = Index.create()
    var tu = index.parse("math_api.h")

    for diagnostic in tu.diagnostics():
        print(diagnostic.format())

    for cursor in tu.cursor():
        if cursor.kind() == CursorKind.FUNCTION_DECL:
            print("function: ", cursor.spelling(), sep="")
        elif cursor.kind() == CursorKind.TYPEDEF_DECL:
            print("typedef: ", cursor.spelling(), sep="")
```

```bash
pixi run mojo run main.mojo
```

## Installation

Install Pixi:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

Create a new Mojo project and add the package channels:

```bash
pixi init clang-demo
cd clang-demo
pixi workspace channel add conda-forge https://conda.modular.com/max https://repo.prefix.dev/modular-community
```

```bash
pixi add mojo libclang_mojo
```

## Developing From Source

Clone this repository and install the development environment:

```bash
git clone https://github.com/MoSafi2/mojo_libclang.git
cd mojo_libclang
pixi install -e dev
```

Run the wrapper test suite through the local shim:

```bash
pixi run -e dev run-test test/test_translation_unit.mojo
```

For build-only checks:

```bash
pixi run -e dev build-test test/_ffi_layout_tests.mojo
```

### Regenerate Low-Level Bindings

Regenerate only when updating LLVM/libclang
coverage or changing ABI handling:

```bash
pixi run -e dev generate
```

Generation updates:

- `clang/_ffi.mojo`
- `test/_ffi_layout_tests.mojo`
- `shim/libclang_mojo_shim.h`
- `shim/libclang_mojo_shim.c`
- `shim/libclang_mojo_shim.so`

Do not edit generated low-level files by hand, either update the generator or
add a deterministic patch.

## Packaging

This repository builds the conda package `libclang_mojo`:

```mojo
from clang.cindex import Index
```

The staged Modular Community recipe lives in
`packaging/modular-community/libclang_mojo/`. To mirror the recipe install
layout locally:

```bash
pixi run -e dev build-package
```

To build the staged recipe for a prefix.dev channel:

```bash
pixi install -e package
PREFIX_CHANNEL=your-channel pixi run -e package render-recipe
PREFIX_CHANNEL=your-channel pixi run -e package build-recipe
```

Upload is explicit and only uploads existing `.conda` artifacts from
`dist/conda/`:

```bash
PREFIX_CHANNEL=your-channel PREFIX_API_KEY=... pixi run -e package upload-recipe
```
