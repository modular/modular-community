# Decimo (formerly DeciMojo) <!-- omit from toc -->

![icon](image.jpeg)

An arbitrary-precision integer and decimal library for Mojo, also with a
128-bit fixed-point decimal type.

## Overview

Decimo does exact base-ten arithmetic, for financial modelling, scientific
computing, and anywhere a floating-point approximation is unacceptable:
`Decimal("0.1") + Decimal("0.2")` is `0.3`, not `0.30000000000000004`.

| Type         | Alias             | Information                                      |
| ------------ | ----------------- | ------------------------------------------------ |
| `BigInt`     | `BInt`            | Equivalent to Python's `int`                     |
| `BigDecimal` | `BDec`, `Decimal` | Equivalent to Python's `decimal.Decimal`         |
| `Decimal128` | `Dec128`          | 128-bit fixed-point, as in C# and Rust           |
| `BigFloat`   | `Float`           | Arbitrary-precision binary float, backed by MPFR |

Beyond the arithmetic there are the exponential, logarithmic and trigonometric
functions, each correctly rounded rather than approximately so, an expression
evaluator (`decimo.eval()`), Chinese numerals, and a TOML parser. `BigInt` is
measured against GMP and `BigDecimal` against libmpdec, the C library behind
Python's `decimal`.

The project also ships an interactive arbitrary-precision calculator, and
publishes the same library to PyPI as a near drop-in for Python's `decimal`.

**Decimo** combines "**Deci**mal" and "**Mo**jo". It is also a Latin word
meaning "tenth", and the root of the word "decimal".

## Documentation

The [README](https://github.com/forfudan/decimo#readme) has a quick start and
the installation instructions; the
[user manual](https://github.com/forfudan/decimo/blob/main/docs/user_manual.md)
is the full tour; the
[changelog](https://github.com/forfudan/decimo/blob/main/docs/changelog.md)
records what each release changed, and
[benchmarks](https://github.com/forfudan/decimo/blob/main/docs/benchmarks.md)
carries the measurements.

## License

Apache License 2.0. See
[LICENSE](https://github.com/forfudan/decimo/blob/main/LICENSE).
