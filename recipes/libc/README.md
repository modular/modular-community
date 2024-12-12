# Mojo's libc

`mojo-libc` is a library that provides access to the C standard library functions in Mojo.

## Getting Started

The only dependency for `libc` is Mojo.

You can install Mojo following the instructions from the [Modular website](https://www.modular.com/max/mojo).

Once you have created a Mojo project using the `magic` tool,

1. Add `libc` as a dependency:
   ```toml
   [dependencies]
   libc = ">=0.1.9"
   ```
2. Run `magic install` at the root of your project, where `mojoproject.toml` is located

3. `libc` should now be installed as a dependency. You can import libc functions from the library, e.g:
    ```mojo
    from libc import socket
    ```

## Supported Functionality

### Basic socket connections

See the examples in [examples/sockets/](https://github.com/msaelices/mojo-libc/tree/main/examples/sockets) directory.

### Basic file system operations

See the examples in [examples/files/](https://github.com/msaelices/mojo-libc/tree/main/examples/files) directory.
