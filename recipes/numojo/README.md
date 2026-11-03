# NuMojo

![logo](./image.jpeg)

[NuMojo](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo) is a library for numerical computing in Mojo 🔥, similar to NumPy in Python.

## About the project

NuMojo aims to encompass the extensive numerics capabilities found in NumPy.

## What NuMojo is

We seek to harness the full potential of Mojo, including vectorization, parallelization, and GPU acceleration. Currently, NuMojo extends most (if not all) standard library math functions to support array inputs.

Our vision for NuMojo is to serve as a familiar and essential building block for other Mojo libraries needing fast math operations, without the additional weight of a machine learning back-propagation system.

## Why NuMojo

- **Native to Mojo.** NuMojo's `NDArray` is a Mojo-native SIMD-backed type, not a binding around NumPy or MAX's tensor types, so it compiles into your program with no Python interop overhead.
- **NumPy-familiar API.** Slicing, broadcasting, `@` for matrix multiplication, and function names mirror NumPy where it makes sense, so existing intuition carries over.
- **Built for Mojo's strengths.** Vectorization and parallelism are used throughout the routines, with GPU and other accelerator support (`AcceleratorNDArray`) landing as Mojo's own device support matures.
