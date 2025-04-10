<br/>
<div align="center">
  <a href="https://github.com/fnands/mimage">
    <img src="image.jpeg" alt="Logo" width="200" height="200">
  </a>

  <h1 align="center">Mimage</h1>

  <p align="center">
    A library for reading images in pure* Mojo 🔥

  ![Language Badge](https://img.shields.io/badge/language-mojo-orange)
  ![CodeQL](https://github.com/fnands/mimage/workflows/CodeQL/badge.svg)
  </p>
</div>

*Not pure Mojo yet, but hopefully soon.
## About The Project

Mimage is a image manipulation library loosely based on Python's [Pillow](https://github.com/python-pillow/Pillow). The goal is to be able to read and write the most popular image formats directly from Mojo.

## Quick Start

Basic usage:

```mojo
import mimage as mi

def main():

  tensor = mi.imread("my/png/image.png")

```

Try out the tests yourself:

```sh
mojo -I .  tests/test_open_png.mojo
```

## Roadmap

### v0.1.0 ✅
- [x] Read simple 8-bit PNGs

### Near term
- [ ] Read jpegs

### Medium term
- [ ] Read more complex PNGs
- [ ] Write PNGs
- [ ] Write jpegs

### Long term
- [ ] v1.0.0 will be achieved when Mimage can open all the same images as Pillow.


## Contributing

Before creating a new issue, please:
* Check if the issue already exists. If an issue is already reported, you can contribute by commenting on the existing issue.
* If not, create a new issue and include all the necessary details to understand/recreate the problem or feature request.

### Creating A Pull Request

1. Fork the Project
2. Create your Feature Branch
3. Commit your Changes
4. Push to the Branch
5. Open a Pull Request
> Once your changes are pushed, navigate to your fork on GitHub. And create a pull request against the original fnands/mimage repository.
> - Before creating a PR make sure it doesn't break any of the unit-tests. (e.g. `mojo -I .  tests/test_open_png.mojo`)
> - Introducing new big features requires a new test!
> - In the pull request, provide a detailed description of the changes and why they're needed. Link any relevant issues.
> - If there are any specific instructions for testing or validating your changes, include those as well.

## License

Distributed under the Apache 2.0 License with LLVM Exceptions. See [LICENSE](https://github.com/fnands/mimage/blob/main/LICENSE) and the LLVM [License](https://llvm.org/LICENSE.txt) for more information.

## Acknowledgements

* Built with [Mojo](https://github.com/modularml/mojo) created by [Modular](https://github.com/modularml)