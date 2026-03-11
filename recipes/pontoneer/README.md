# Pontoneer

![icon](image.jpeg)

Pontoneer is a Mojo library that provides an extension to the Python extension capabilities provided by the standard library. Pontoneer adds support for:

- mapping protocol
- number protocol
- sequence protocol
- rich comparison in the type protocol

## Installation

Pontoneer requires the nightly Mojo and uses [pixi](https://pixi.sh) for environment management.

```bash
git clone https://github.com/winding-lines/pontoneer.git
cd pontoneer
pixi install
```

To install in your own application as a library

```bash
pixi add --channel https://repo.prefix.dev/modular-community --channel https://conda.modular.com/max-nightly pontoneer
```
```
```

## Quick Start

An example can be found in https://github.com/winding-lines/pontoneer/tree/main/examples/columnar.


## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
