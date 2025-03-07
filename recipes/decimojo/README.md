# DeciMojo

A correctly-rounded, fixed-point decimal arithmetic library for the [Mojo programming language 🔥](https://www.modular.com/mojo).

## Overview

DeciMojo provides a Decimal type implementation for Mojo with fixed-precision arithmetic, designed to handle financial calculations and other scenarios where floating-point rounding errors are problematic.

## Objective

Financial calculations and data analysis require precise decimal arithmetic that floating-point numbers cannot reliably provide. As someone working in finance and credit risk model validation, I needed a dependable fixed-precision numeric type when migrating my personal projects from Python to Mojo.

Since Mojo currently lacks a native Decimal type in its standard library, I decided to create my own implementation to fill that gap.

This project draws inspiration from several established decimal implementations and documentation, e.g., [Python built-in `Decimal` type](https://docs.python.org/3/library/decimal.html), [Rust `rust_decimal` crate](https://docs.rs/rust_decimal/latest/rust_decimal/index.html), [Microsoft's `Decimal` implementation](https://learn.microsoft.com/en-us/dotnet/api/system.decimal.getbits?view=net-9.0&redirectedfrom=MSDN#System_Decimal_GetBits_System_Decimal_), [General Decimal Arithmetic Specification](https://speleotrove.com/decimal/decarith.html), etc. Many thanks to these predecessors for their contributions and their commitment to open knowledge sharing.

## Status

Rome is not built in one day. DeciMojo is currently under active development and appears to be between the **"make it work"** and **"make it right"** phases, leaning more toward the latter. Bug reports and feature requests are welcome! If you encounter issues, please [file them here](https://github.com/forFudan/decimojo/issues).

### Make it Work ✅ (MOSTLY COMPLETED)

- Core decimal implementation exists and functions
- Basic arithmetic operations (+, -, *, /) are implemented
- Type conversions to/from various formats work
- String representation and parsing are functional
- Construction from different sources (strings, numbers) is supported

### Make it Right 🔄 (IN PROGRESS)

- Edge case handling is being addressed (division by zero, zero to negative power)
- Scale and precision management shows sophistication
- Financial calculations demonstrate proper rounding
- High precision support is implemented (up to 28 decimal places)
- The examples show robust handling of various scenarios

### Make it Fast ⏳ (FUTURE WORK)

- Performance optimization is acknowledged but not currently prioritized

## Examples

Here are 10 key examples highlighting the most important features of the `Decimal` type in its current state:

### 1. Fixed-Point Precision for Financial Calculations

```mojo
from decimojo.prelude import *

# The classic floating-point problem
print(0.1 + 0.2)  # 0.30000000000000004 (not exactly 0.3)

# Decimal solves this with exact representation
var d1 = Decimal("0.1")
var d2 = Decimal("0.2")
var sum = d1 + d2
print(sum)  # Exactly 0.3

# Financial calculation example - computing tax
var price = Decimal("19.99")
var tax_rate = Decimal("0.0725")
var tax = price * tax_rate  # Exactly 1.449275
var total = price + tax     # Exactly 21.439275
```

### 2. Basic Arithmetic with Proper Rounding

```mojo
# Addition with different scales
var a = Decimal("123.45")
var b = Decimal("67.8")
print(a + b)  # 191.25 (preserves highest precision)

# Subtraction with negative result
var c = Decimal("50")
var d = Decimal("75.25")
print(c - d)  # -25.25

# Multiplication preserving full precision
var e = Decimal("12.34")
var f = Decimal("5.67")
print(e * f)  # 69.9678 (all digits preserved)

# Division with repeating decimals handled precisely
var g = Decimal("1")
var h = Decimal("3")
print(g / h)  # 0.3333333333333333333333333333 (to precision limit)
```

### 3. Scale and Precision Management

```mojo
# Scale refers to number of decimal places
var d1 = Decimal("123.45")
print(d1.scale())  # 2

var d2 = Decimal("123.4500")
print(d2.scale())  # 4

# Arithmetic operations combine scales appropriately
var sum = Decimal("0.123") + Decimal("0.45")  # Takes larger scale
print(sum)  # 0.573

var product = Decimal("0.12") * Decimal("0.34")  # Sums the scales
print(product)  # 0.0408

# High precision is preserved (up to 28 decimal places)
var precise = Decimal("0.1234567890123456789012345678")
print(precise)  # 0.1234567890123456789012345678
```

### 4. Sign Handling and Absolute Value

```mojo
# Negation operator
var pos = Decimal("123.45")
var neg = -pos
print(neg)  # -123.45

# Multiple negations
var back_to_pos = -(-pos)
print(back_to_pos)  # 123.45

# Absolute value
var abs_val = abs(Decimal("-987.65"))
print(abs_val)  # 987.65

# Sign checking
print(Decimal("-123.45").is_negative())  # True
print(Decimal("0").is_negative())        # False
print(Decimal("123.45").is_negative())   # False
```

### 5. Advanced Mathematical Operations

```mojo
from decimojo.mathematics import sqrt

# Integer powers
var squared = Decimal("3") ** 2
print(squared)  # 9

# Negative powers (reciprocals)
var recip = Decimal("2") ** (-1)
print(recip)  # 0.5

# Square root with high precision
var root2 = sqrt(Decimal("2"))
print(root2)  # 1.414213562373095048801688724...

# Special cases
print(Decimal("10") ** 0)  # 1 (anything to power 0)
print(Decimal("1") ** 20)  # 1 (1 to any power)
print(Decimal("0") ** 5)   # 0 (0 to positive power)
```

### 6. Type Conversions and Interoperability

```mojo
var d = Decimal("123.456")

# Converting to string (for display or serialization)
var str_val = String(d)
print(str_val)  # "123.456"

# Getting internal representation 
print(repr(d))  # Shows internal state

# Converting to int (truncates toward zero)
var i = Int(d)
print(i)  # 123

# Converting to float (may lose precision)
var f = Float64(d)
print(f)  # 123.456
```

### 7. Handling Edge Cases and Errors

```mojo
# Division by zero is properly caught
try:
    var result = Decimal("10") / Decimal("0")
except:
    print("Division by zero properly detected")

# Zero raised to negative power
try:
    var invalid = Decimal("0") ** (-1)
except:
    print("Zero to negative power properly detected")
    
# Smallest representable positive value
var tiny = Decimal("0." + "0" * 27 + "1")  # 28 decimal places
print(tiny)  # 0.0000000000000000000000000001
```

### 8. Equality and Zero Comparisons

```mojo
# Equal values with different representations
var a = Decimal("123.4500")
var b = Decimal("123.45")
print(a == b)  # True (numeric value equality)

# Zero values with different scales
var z1 = Decimal("0")
var z2 = Decimal("0.000")
print(z1 == z2)  # True

# Zero detection
print(z1.is_zero())  # True
print(z2.is_zero())  # True
```

### 9. Real World Financial Examples

```mojo
# Monthly loan payment calculation
var principal = Decimal("200000")  # $200,000 loan
var annual_rate = Decimal("0.05")  # 5% interest rate
var monthly_rate = annual_rate / Decimal("12")
var num_payments = Decimal("360")  # 30 years

# Monthly payment formula: P * r(1+r)^n/((1+r)^n-1)
var numerator = monthly_rate * (Decimal("1") + monthly_rate) ** 360
var denominator = (Decimal("1") + monthly_rate) ** 360 - Decimal("1")
var payment = principal * (numerator / denominator)
print("Monthly payment: $" + String(round(payment, 2)))  # $1,073.64

# Correct handling of multiple items and discounts
var item1 = Decimal("29.99")
var item2 = Decimal("59.99")
var subtotal = item1 + item2  # 89.98
var discount = subtotal * Decimal("0.15")  # 15% off
var after_discount = subtotal - discount
var tax = after_discount * Decimal("0.08")  # 8% tax
var final = after_discount + tax
print("Final price: $" + String(round(final, 2)))
```

### 10. Maximum Precision and Limit Testing

```mojo
# Maximum value supported
var max_val = Decimal.MAX()
print(max_val)  # 79228162514264337593543950335

# Minimum value supported
var min_val = Decimal.MIN()
print(min_val)  # -79228162514264337593543950335

# Operations near limits
var near_max = Decimal("79228162514264337593543950334")  # MAX() - 1
var still_valid = near_max + Decimal("1")
print(still_valid == max_val)  # True

# Maximum precision for high-accuracy scientific calculations
var pi = Decimal("3.1415926535897932384626433832")
var radius = Decimal("2.5")
var area = pi * (radius ** 2)
print("Circle area: " + String(area))  # Precisely calculated area
```

## Tests and benches

After cloning the repo onto your local disk, you can:

- Use `magic run test` (or `maigic run t`) to run tests.
- Use `magic run bench` (or `magic run b`) to generate logs for benchmarking tests agains `python.decimal` module. The log files are saved in `benches/logs/`.

## Related Projects

I am also working on NuMojo, a library for numerical computing in Mojo 🔥 similar to NumPy, SciPy in Python. If you are also interested, you can [check it out here](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo).

## License

Distributed under the Apache 2.0 License. See [LICENSE](https://github.com/forFudan/decimojo/blob/main/LICENSE) for details.

## Acknowledgements

Built with the [Mojo programming language 🔥](https://www.modular.com/mojo) created by [Modular](https://www.modular.com/).
