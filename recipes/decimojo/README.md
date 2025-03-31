# DeciMojo

![icon](icon_256x256.png)

A comprehensive decimal and integer mathematics library for [Mojo](https://www.modular.com/mojo).

**[中文·漢字»](https://zhuyuhao.com/decimojo/docs/readme_zht.html)**　|　**[Repository on GitHub»](https://github.com/forfudan/decimojo)**

## Overview

DeciMojo provides a comprehensive decimal and integer mathematics library for Mojo, delivering exact calculations for financial modeling, scientific computing, and applications where floating-point approximation errors are unacceptable. Beyond basic arithmetic, the library includes advanced mathematical functions with guaranteed precision.

The core types are:

- A 128-bit fixed-point decimal implementation (`Decimal`) supporting up to 29 significant digits with a maximum of 28 decimal places[^fixed]. It features a complete set of mathematical functions including logarithms, exponentiation, roots, and trigonometric operations.
- A base-10 arbitrary-precision signed integer type (`BigInt`) and a base-10 arbitrary-precision unsigned integer type (`BigUInt`) supporting unlimited digits[^integer]. It features comprehensive arithmetic operations, comparison functions, and supports extremely large integer calculations efficiently.

The library is expanding to include `BigDecimal` types that support arbitrary precision[^arbitrary], allowing for calculations with unlimited digits and decimal places. These extensions are currently under active development.

## Installation

DeciMojo is available in the [modular-community](https://repo.prefix.dev/modular-community) package repository. You can install it using any of these methods:

From the `magic` CLI, simply run ```magic add decimojo```. This fetches the latest version and makes it immediately available for import.

For projects with a `mojoproject.toml`file, add the dependency ```decimojo = ">=0.2.0"```. Then run `magic install` to download and install the package.

For the latest development version, clone the [GitHub repository](https://github.com/forfudan/decimojo) and build the package locally.

| `decimojo` | `mojo` |
| ---------- | ------ |
| v0.1.0     | >=25.1 |
| v0.2.0     | >=25.2 |

## Quick start

Here is a comprehensive quick-start guide showcasing each major function of the `Decimal` type.

```mojo
from decimojo import Decimal, RoundingMode

fn main() raises:
    # === Construction ===
    var a = Decimal("123.45")                        # From string
    var b = Decimal(123)                             # From integer
    var c = Decimal(123, 2)                          # Integer with scale (1.23)
    var d = Decimal.from_float(3.14159)              # From floating-point
    
    # === Basic Arithmetic ===
    print(a + b)                                     # Addition: 246.45
    print(a - b)                                     # Subtraction: 0.45
    print(a * b)                                     # Multiplication: 15184.35
    print(a / b)                                     # Division: 1.0036585365853658536585365854
    
    # === Rounding & Precision ===
    print(a.round(1))                                # Round to 1 decimal place: 123.5
    print(a.quantize(Decimal("0.01")))               # Format to 2 decimal places: 123.45
    print(a.round(0, RoundingMode.ROUND_DOWN))       # Round down to integer: 123
    
    # === Comparison ===
    print(a > b)                                     # Greater than: True
    print(a == Decimal("123.45"))                    # Equality: True
    print(a.is_zero())                               # Check for zero: False
    print(Decimal("0").is_zero())                    # Check for zero: True
    
    # === Type Conversions ===
    print(Float64(a))                                # To float: 123.45
    print(a.to_int())                                # To integer: 123
    print(a.to_str())                                # To string: "123.45"
    print(a.coefficient())                           # Get coefficient: 12345
    print(a.scale())                                 # Get scale: 2
    
    # === Mathematical Functions ===
    print(Decimal("2").sqrt())                       # Square root: 1.4142135623730950488016887242
    print(Decimal("100").root(3))                    # Cube root: 4.641588833612778892410076351
    print(Decimal("2.71828").ln())                   # Natural log: 0.9999993273472820031578910056
    print(Decimal("10").log10())                     # Base-10 log: 1
    print(Decimal("16").log(Decimal("2")))           # Log base 2: 3.9999999999999999999999999999
    print(Decimal("10").exp())                       # e^10: 22026.465794806716516957900645
    print(Decimal("2").power(10))                    # Power: 1024
    
    # === Sign Handling ===
    print(-a)                                        # Negation: -123.45
    print(abs(Decimal("-123.45")))                   # Absolute value: 123.45
    print(Decimal("123.45").is_negative())           # Check if negative: False
    
    # === Special Values ===
    print(Decimal.PI())                              # π constant: 3.1415926535897932384626433833
    print(Decimal.E())                               # e constant: 2.7182818284590452353602874714
    print(Decimal.ONE())                             # Value 1: 1
    print(Decimal.ZERO())                            # Value 0: 0
    print(Decimal.MAX())                             # Maximum value: 79228162514264337593543950335
    
    # === Convenience Methods ===
    print(Decimal("123.400").is_integer())           # Check if integer: False
    print(a.number_of_significant_digits())          # Count significant digits: 5
    print(Decimal("12.34").to_str_scientific())      # Scientific notation: 1.234E+1
```

[Click here for 8 key examples](https://zhuyuhao.com/decimojo/docs/examples) highlighting the most important features of the `Decimal` type.

Here is a comprehensive quick-start guide showcasing each major function of the `BigInt` type.

```mojo
from decimojo import BigInt, BInt
# BInt is an alias for BigInt

fn main() raises:
    # === Construction ===
    var a = BigInt("12345678901234567890")         # From string
    var b = BInt(12345)                            # From integer
    
    # === Basic Arithmetic ===
    print(a + b)                                   # Addition: 12345678901234580235
    print(a - b)                                   # Subtraction: 12345678901234555545
    print(a * b)                                   # Multiplication: 152415787814108380241050
    
    # === Division Operations ===
    print(a // b)                                  # Floor division: 999650944609516
    print(a.truncate_divide(b))                    # Truncate division: 999650944609516
    print(a % b)                                   # Modulo: 9615
    
    # === Power Operation ===
    print(BInt(2).power(10))                     # Power: 1024
    print(BInt(2) ** 10)                         # Power (using ** operator): 1024
    
    # === Comparison ===
    print(a > b)                                   # Greater than: True
    print(a == BInt("12345678901234567890"))     # Equality: True
    print(a.is_zero())                             # Check for zero: False
    
    # === Type Conversions ===
    print(a.to_str())                              # To string: "12345678901234567890"
    
    # === Sign Handling ===
    print(-a)                                      # Negation: -12345678901234567890
    print(abs(BInt("-12345678901234567890")))    # Absolute value: 12345678901234567890
    print(a.is_negative())                         # Check if negative: False

    # === Extremely large numbers ===
    # 3600 digits // 1800 digits
    print(BInt("123456789" * 400) // BInt("987654321" * 200))
```

## Objective

Financial calculations and data analysis require precise decimal arithmetic that floating-point numbers cannot reliably provide. As someone working in finance and credit risk model validation, I needed a dependable correctly-rounded, fixed-precision numeric type when migrating my personal projects from Python to Mojo.

Since Mojo currently lacks a native Decimal type in its standard library, I decided to create my own implementation to fill that gap.

This project draws inspiration from several established decimal implementations and documentation, e.g., [Python built-in `Decimal` type](https://docs.python.org/3/library/decimal.html), [Rust `rust_decimal` crate](https://docs.rs/rust_decimal/latest/rust_decimal/index.html), [Microsoft's `Decimal` implementation](https://learn.microsoft.com/en-us/dotnet/api/system.decimal.getbits?view=net-9.0&redirectedfrom=MSDN#System_Decimal_GetBits_System_Decimal_), [General Decimal Arithmetic Specification](https://speleotrove.com/decimal/decarith.html), etc. Many thanks to these predecessors for their contributions and their commitment to open knowledge sharing.

## Nomenclature

DeciMojo combines "Deci" and "Mojo" - reflecting its purpose and implementation language. "Deci" (from Latin "decimus" meaning "tenth") highlights our focus on the decimal numeral system that humans naturally use for counting and calculations.

Although the name emphasizes decimals with fractional parts, DeciMojo embraces the full spectrum of decimal mathematics. Our `BigInt` type, while handling only integers, is designed specifically for the decimal numeral system with its base-10 internal representation. This approach offers optimal performance while maintaining human-readable decimal semantics, contrasting with binary-focused libraries. Furthermore, `BigInt` serves as the foundation for our `BigDecimal` implementation, enabling arbitrary-precision calculations across both integer and fractional domains.

The name ultimately emphasizes our mission: bringing precise, reliable decimal calculations to the Mojo ecosystem, addressing the fundamental need for exact arithmetic that floating-point representations cannot provide.

## Status

Rome wasn't built in a day. DeciMojo is currently under active development. For the 128-bit `Decimal` type, it has successfully progressed through the **"make it work"** phase and is now well into the **"make it right"** phase with many optimizations already in place. Bug reports and feature requests are welcome! If you encounter issues, please [file them here](https://github.com/forfudan/decimojo/issues).

### Make it Work ✅ (COMPLETED)

- Core decimal implementation with a robust 128-bit representation (96-bit coefficient + 32-bit flags)
- Comprehensive arithmetic operations (+, -, *, /, %, **) with proper overflow handling
- Type conversions to/from various formats (String, Int, Float64, etc.)
- Proper representation of special values (NaN, Infinity)
- Full suite of comparison operators with correct decimal semantics

### Make it Right 🔄 (MOSTLY COMPLETED)

- Reorganized codebase with modular structure (decimal, arithmetics, comparison, exponential).
- Edge case handling for all operations (division by zero, zero to negative power).
- Scale and precision management with sophisticated rounding strategies.
- Financial calculations with banker's rounding (ROUND_HALF_EVEN).
- High-precision advanced mathematical functions (sqrt, root, ln, exp, log10, power).
- Proper implementation of traits (Absable, Comparable, Floatable, Roundable, etc).
- **BigInt and BigUInt** implementations with complete arithmetic operations, proper division semantics (floor and truncate), and support for arbitrary-precision calculations.

### Make it Fast ⚡ (SIGNIFICANT PROGRESS)

DeciMojo delivers exceptional performance compared to Python's `decimal` module while maintaining precise calculations. This performance difference stems from fundamental design choices:

- **DeciMojo**: Uses a fixed 128-bit representation (96-bit coefficient + 32-bit flags) with a maximum of 28 decimal places, optimized for modern hardware and Mojo's performance capabilities.
- **Python decimal**: Implements arbitrary precision that can represent numbers with unlimited significant digits but requires dynamic memory allocation and more complex algorithms.

This architectural difference explains our benchmarking results:

- Core arithmetic operations (+, -, *, /) achieve 100x-3500x speedup over Python's decimal module.
- Special case handling (powers of 0, 1, etc.) shows up to 3500x performance improvement.
- Advanced mathematical functions (sqrt, ln, exp) demonstrate 5x-600x better performance.
- Only specific edge cases (like computing 10^(1/100)) occasionally perform better in Python due to its arbitrary precision algorithms.

Regular benchmarks against Python's `decimal` module are available in the `bench/` folder, documenting both the performance advantages and the few specific operations where different approaches are needed.

### Future Extensions 🚀 (PLANNED)

- **BigDecimal**: 🔄 **IN PROGRESS** - Arbitrary-precision decimal type with configurable precision[^arbitrary].
- **BigComplex**: 📝 **PLANNED** - Arbitrary-precision complex number type built on BigDecimal.

## Tests and benches

After cloning the repo onto your local disk, you can:

- Use `magic run test` to run tests.
- Use `magic run bench_decimal` to generate logs for benchmarking tests against `python.decimal` module. The log files are saved in `benches/decimal/logs/`.

## Citation

If you find DeciMojo useful for your research, consider listing it in your citations.

```tex
@software{Zhu.2025,
    author       = {Zhu, Yuhao},
    year         = {2025},
    title        = {A comprehensive decimal and integer mathematics library for Mojo},
    url          = {https://github.com/forfudan/decimojo},
    version      = {0.2.0},
    note         = {Computer Software}
}
```

## License

This repository and its contributions are licensed under the Apache License v2.0.

[^fixed]: The `Decimal` type can represent values with up to 29 significant digits and a maximum of 28 digits after the decimal point. When a value exceeds the maximum representable value (`2^96 - 1`), DeciMojo either raises an error or rounds the value to fit within these constraints. For example, the significant digits of `8.8888888888888888888888888888` (29 eights total with 28 after the decimal point) exceeds the maximum representable value (`2^96 - 1`) and is automatically rounded to `8.888888888888888888888888889` (28 eights total with 27 after the decimal point). DeciMojo's `Decimal` type is similar to `System.Decimal` (C#/.NET), `rust_decimal` in Rust, `DECIMAL/NUMERIC` in SQL Server, etc.
[^integer]: The BigInt implementation uses a base-10 representation for users (maintaining decimal semantics), while internally using an optimized base-10^9 storage system for efficient calculations. This approach balances human-readable decimal operations with high-performance computing. It provides both floor division (round toward negative infinity) and truncate division (round toward zero) semantics, enabling precise handling of division operations with correct mathematical behavior regardless of operand signs.
[^arbitrary]: Built on top of our completed BigInt implementation, BigDecimal will support arbitrary precision for both the integer and fractional parts, similar to `decimal` and `mpmath` in Python, `java.math.BigDecimal` in Java, etc.
