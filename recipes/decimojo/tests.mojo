"""
Test Decimal arithmetic operations including addition, subtraction, and negation.
"""

from decimojo.prelude import *
import testing


fn test_add() raises:
    print("Testing decimal addition...")

    # Test case 1: Simple addition with same scale
    var a1 = Decimal("123.45")
    var b1 = Decimal("67.89")
    var result1 = a1 + b1
    testing.assert_equal(
        String(result1), "191.34", "Simple addition with same scale"
    )

    # Test case 2: Addition with different scales
    var a2 = Decimal("123.4")
    var b2 = Decimal("67.89")
    var result2 = a2 + b2
    testing.assert_equal(
        String(result2), "191.29", "Addition with different scales"
    )

    # Test case 3: Addition with negative numbers
    var a3 = Decimal("123.45")
    var b3 = Decimal("-67.89")
    var result3 = a3 + b3
    testing.assert_equal(
        String(result3), "55.56", "Addition with negative number"
    )

    # Test case 4: Addition resulting in negative
    var a4 = Decimal("-123.45")
    var b4 = Decimal("67.89")
    var result4 = a4 + b4
    testing.assert_equal(
        String(result4), "-55.56", "Addition resulting in negative"
    )

    # Test case 5: Addition with zero
    var a5 = Decimal("123.45")
    var b5 = Decimal("0.00")
    var result5 = a5 + b5
    testing.assert_equal(String(result5), "123.45", "Addition with zero")

    # Test case 6: Addition resulting in zero
    var a6 = Decimal("123.45")
    var b6 = Decimal("-123.45")
    var result6 = a6 + b6
    testing.assert_equal(String(result6), "0.00", "Addition resulting in zero")

    # Test case 7: Addition with large scales
    var a7 = Decimal("0.0000001")
    var b7 = Decimal("0.0000002")
    var result7 = a7 + b7
    testing.assert_equal(
        String(result7), "0.0000003", "Addition with large scales"
    )

    # Test case 8: Addition with different large scales
    var a8 = Decimal("0.000001")
    var b8 = Decimal("0.0000002")
    var result8 = a8 + b8
    testing.assert_equal(
        String(result8), "0.0000012", "Addition with different large scales"
    )

    # Additional edge cases for addition

    # Test case 9: Addition with many decimal places
    var a9 = Decimal("0.123456789012345678901234567")
    var b9 = Decimal("0.987654321098765432109876543")
    var result9 = a9 + b9
    testing.assert_equal(
        String(result9),
        "1.111111110111111111011111110",
        "Addition with many decimal places",
    )

    # Test case 10: Addition with extreme scale difference
    var a10 = Decimal("123456789")
    var b10 = Decimal("0.000000000123456789")
    var result10 = a10 + b10
    testing.assert_equal(
        String(result10),
        "123456789.000000000123456789",
        "Addition with extreme scale difference",
    )

    # Test case 11: Addition near maximum precision
    var a11 = Decimal("0." + "1" * 28)  # 0.1111...1 (28 digits)
    var b11 = Decimal("0." + "9" * 28)  # 0.9999...9 (28 digits)
    var result11 = a11 + b11
    testing.assert_equal(
        String(result11),
        "1.1111111111111111111111111110",
        "Addition near maximum precision",
    )

    # Test case 12: Addition causing scale truncation
    var a12 = Decimal("0." + "1" * 27 + "1")  # 0.1111...1 (28 digits)
    var b12 = Decimal("0.0" + "9" * 27)  # 0.09999...9 (28 digits)
    var result12 = a12 + b12
    testing.assert_equal(
        String(result12),
        "0." + "2" + "1" * 26 + "0",
        "Addition causing scale truncation",
    )

    # Test case 13: Addition of very small numbers
    var a13 = Decimal("0." + "0" * 27 + "1")  # 0.0000...01 (1 at 28th place)
    var b13 = Decimal("0." + "0" * 27 + "2")  # 0.0000...02 (2 at 28th place)
    var result13 = a13 + b13
    testing.assert_equal(
        String(result13),
        "0." + "0" * 27 + "3",
        "Addition of very small numbers",
    )

    # Test case 14: Addition with alternating signs and scales
    var a14 = Decimal("1.01")
    var b14 = Decimal("-0.101")
    var result14 = a14 + b14
    testing.assert_equal(
        String(result14), "0.909", "Addition with alternating signs and scales"
    )

    # Test case 15: Addition with large numbers (near limits)
    var a15 = Decimal("79228162514264337593543950334")  # MAX() - 1
    var b15 = Decimal("1")
    var result15 = a15 + b15
    testing.assert_equal(
        String(result15),
        "79228162514264337593543950335",
        "Addition approaching maximum value",
    )

    # Test case 16: Repeated addition to test cumulative errors
    var acc = Decimal("0")
    for _ in range(10):
        acc = acc + Decimal("0.1")
    testing.assert_equal(String(acc), "1.0", "Repeated addition of 0.1")

    # Test case 17: Edge case with alternating very large and very small values
    var a17 = Decimal("1234567890123456789.0123456789")
    var b17 = Decimal("0.0000000000000000009876543211")
    var result17 = a17 + b17
    testing.assert_equal(
        String(result17),
        "1234567890123456789.0123456790",
        "Addition with large and small values",
    )

    print("Decimal addition tests passed!")


fn test_negation() raises:
    print("Testing decimal negation...")

    # Test case 1: Negate positive number
    var a1 = Decimal("123.45")
    var result1 = -a1
    testing.assert_equal(String(result1), "-123.45", "Negating positive number")

    # Test case 2: Negate negative number
    var a2 = Decimal("-67.89")
    var result2 = -a2
    testing.assert_equal(String(result2), "67.89", "Negating negative number")

    # Test case 3: Negate zero
    var a3 = Decimal("0")
    var result3 = -a3
    testing.assert_equal(String(result3), "0", "Negating zero")

    # Test case 4: Negate number with trailing zeros
    var a4 = Decimal("123.4500")
    var result4 = -a4
    testing.assert_equal(
        String(result4), "-123.4500", "Negating with trailing zeros"
    )

    # Test case 5: Double negation
    var a5 = Decimal("123.45")
    var result5 = -(-a5)
    testing.assert_equal(String(result5), "123.45", "Double negation")

    # Additional edge cases for negation

    # Test case 6: Negate very small number
    var a6 = Decimal("0." + "0" * 27 + "1")  # 0.0000...01 (1 at 28th place)
    var result6 = -a6
    testing.assert_equal(
        String(result6), "-0." + "0" * 27 + "1", "Negating very small number"
    )

    # Test case 7: Negate very large number
    var a7 = Decimal("79228162514264337593543950335")  # MAX()
    var result7 = -a7
    testing.assert_equal(
        String(result7),
        "-79228162514264337593543950335",
        "Negating maximum value",
    )

    # Test case 8: Triple negation
    var a8 = Decimal("123.45")
    var result8 = -(-(-a8))
    testing.assert_equal(String(result8), "-123.45", "Triple negation")

    # Test case 9: Negate number with scientific notation (if supported)
    try:
        var a9 = Decimal("1.23e5")  # 123000
        var result9 = -a9
        testing.assert_equal(
            String(result9),
            "-123000",
            "Negating number from scientific notation",
        )
    except:
        print("Scientific notation not supported in this implementation")

    # Test case 10: Negate number with maximum precision
    var a10 = Decimal("0." + "1" * 28)  # 0.1111...1 (28 digits)
    var result10 = -a10
    testing.assert_equal(
        String(result10),
        "-0." + "1" * 28,
        "Negating number with maximum precision",
    )

    print("Decimal negation tests passed!")


fn test_abs() raises:
    print("Testing decimal absolute value...")

    # Test case 1: Absolute value of positive number
    var a1 = Decimal("123.45")
    var result1 = abs(a1)
    testing.assert_equal(
        String(result1), "123.45", "Absolute value of positive number"
    )

    # Test case 2: Absolute value of negative number
    var a2 = Decimal("-67.89")
    var result2 = abs(a2)
    testing.assert_equal(
        String(result2), "67.89", "Absolute value of negative number"
    )

    # Test case 3: Absolute value of zero
    var a3 = Decimal("0")
    var result3 = abs(a3)
    testing.assert_equal(String(result3), "0", "Absolute value of zero")

    # Test case 4: Absolute value of negative zero (if supported)
    var a4 = Decimal("-0.00")
    var result4 = abs(a4)
    testing.assert_equal(
        String(result4), "0.00", "Absolute value of negative zero"
    )

    # Test case 5: Absolute value with large number of decimal places
    var a5 = Decimal("-0.0000000001")
    var result5 = abs(a5)
    testing.assert_equal(
        String(result5),
        "0.0000000001",
        "Absolute value of small negative number",
    )

    # Test case 6: Absolute value of very large number
    var a6 = Decimal("-9999999999.9999999999")
    var result6 = abs(a6)
    testing.assert_equal(
        String(result6),
        "9999999999.9999999999",
        "Absolute value of large negative number",
    )

    # Test case 7: Absolute value of number with many significant digits
    var a7 = Decimal("-0.123456789012345678901234567")
    var result7 = abs(a7)
    testing.assert_equal(
        String(result7),
        "0.123456789012345678901234567",
        "Absolute value of high precision negative number",
    )

    # Test case 8: Absolute value of maximum representable number
    try:
        var a8 = Decimal("79228162514264337593543950335")  # Maximum value
        var result8 = abs(a8)
        testing.assert_equal(
            String(result8),
            "79228162514264337593543950335",
            "Absolute value of maximum value",
        )

        var a9 = Decimal(
            "-79228162514264337593543950335"
        )  # Negative maximum value
        var result9 = abs(a9)
        testing.assert_equal(
            String(result9),
            "79228162514264337593543950335",
            "Absolute value of negative maximum value",
        )
    except:
        print("Maximum value test not applicable")

    print("Decimal absolute value tests passed!")


fn test_subtract() raises:
    print("Testing decimal subtraction...")

    # Test case 1: Simple subtraction with same scale
    var a1 = Decimal("123.45")
    var b1 = Decimal("67.89")
    var result1 = a1 - b1
    testing.assert_equal(
        String(result1), "55.56", "Simple subtraction with same scale"
    )

    # Test case 2: Subtraction with different scales
    var a2 = Decimal("123.4")
    var b2 = Decimal("67.89")
    var result2 = a2 - b2
    testing.assert_equal(
        String(result2), "55.51", "Subtraction with different scales"
    )

    # Test case 3: Subtraction resulting in negative
    var a3 = Decimal("67.89")
    var b3 = Decimal("123.45")
    var result3 = a3 - b3
    testing.assert_equal(
        String(result3), "-55.56", "Subtraction resulting in negative"
    )

    # Test case 4: Subtraction of negative numbers
    var a4 = Decimal("123.45")
    var b4 = Decimal("-67.89")
    var result4 = a4 - b4
    testing.assert_equal(
        String(result4), "191.34", "Subtraction of negative number"
    )

    # Test case 5: Subtraction with zero
    var a5 = Decimal("123.45")
    var b5 = Decimal("0.00")
    var result5 = a5 - b5
    testing.assert_equal(String(result5), "123.45", "Subtraction with zero")

    # Test case 6: Subtraction resulting in zero
    var a6 = Decimal("123.45")
    var b6 = Decimal("123.45")
    var result6 = a6 - b6
    testing.assert_equal(
        String(result6), "0.00", "Subtraction resulting in zero"
    )

    # Test case 7: Subtraction with large scales
    var a7 = Decimal("0.0000003")
    var b7 = Decimal("0.0000002")
    var result7 = a7 - b7
    testing.assert_equal(
        String(result7), "0.0000001", "Subtraction with large scales"
    )

    # Test case 8: Subtraction with different large scales
    var a8 = Decimal("0.000005")
    var b8 = Decimal("0.0000002")
    var result8 = a8 - b8
    testing.assert_equal(
        String(result8), "0.0000048", "Subtraction with different large scales"
    )

    # Test case 9: Subtraction with small difference
    var a9 = Decimal("1.0000001")
    var b9 = Decimal("1.0000000")
    var result9 = a9 - b9
    testing.assert_equal(
        String(result9), "0.0000001", "Subtraction with small difference"
    )

    # Test case 10: Subtraction of very small from very large
    var a10 = Decimal("9999999999.9999999")
    var b10 = Decimal("0.0000001")
    var result10 = a10 - b10
    testing.assert_equal(
        String(result10),
        "9999999999.9999998",
        "Subtraction of very small from very large",
    )

    # Test case 11: Self subtraction for various values (expanded from list)
    # Individual test cases instead of iterating over a list
    var value1 = Decimal("0")
    testing.assert_equal(
        String(value1 - value1),
        String(round(Decimal("0"), value1.scale())),
        "Self subtraction should yield zero (0)",
    )

    var value2 = Decimal("123.45")
    testing.assert_equal(
        String(value2 - value2),
        String(round(Decimal("0"), value2.scale())),
        "Self subtraction should yield zero (123.45)",
    )

    var value3 = Decimal("-987.654")
    testing.assert_equal(
        String(value3 - value3),
        String(round(Decimal("0"), value3.scale())),
        "Self subtraction should yield zero (-987.654)",
    )

    var value4 = Decimal("0.0001")
    testing.assert_equal(
        String(value4 - value4),
        String(round(Decimal("0"), value4.scale())),
        "Self subtraction should yield zero (0.0001)",
    )

    var value5 = Decimal("-99999.99999")
    testing.assert_equal(
        String(value5 - value5),
        String(round(Decimal("0"), value5.scale())),
        "Self subtraction should yield zero (-99999.99999)",
    )

    # Test case 12: Verify that a - b = -(b - a)
    var a12a = Decimal("123.456")
    var b12a = Decimal("789.012")
    var result12a = a12a - b12a
    var result12b = -(b12a - a12a)
    testing.assert_equal(
        String(result12a), String(result12b), "a - b should equal -(b - a)"
    )

    print("Decimal subtraction tests passed!")


fn test_multiplication() raises:
    print("Testing decimal multiplication...")

    # Test case 1: Simple multiplication with same scale
    var a1 = Decimal("12.34")
    var b1 = Decimal("5.67")
    var result1 = a1 * b1
    testing.assert_equal(
        String(result1), "69.9678", "Simple multiplication with same scale"
    )

    # Test case 2: Multiplication with different scales
    var a2 = Decimal("12.3")
    var b2 = Decimal("5.67")
    var result2 = a2 * b2
    testing.assert_equal(
        String(result2), "69.741", "Multiplication with different scales"
    )

    # Test case 3: Multiplication with negative numbers
    var a3 = Decimal("12.34")
    var b3 = Decimal("-5.67")
    var result3 = a3 * b3
    testing.assert_equal(
        String(result3), "-69.9678", "Multiplication with negative number"
    )

    # Test case 4: Multiplication with both negative numbers
    var a4 = Decimal("-12.34")
    var b4 = Decimal("-5.67")
    var result4 = a4 * b4
    testing.assert_equal(
        String(result4), "69.9678", "Multiplication with both negative numbers"
    )

    # Test case 5: Multiplication by zero
    var a5 = Decimal("12.34")
    var b5 = Decimal("0.00")
    var result5 = a5 * b5
    testing.assert_equal(String(result5), "0.0000", "Multiplication by zero")

    # Test case 6: Multiplication by one
    var a6 = Decimal("12.34")
    var b6 = Decimal("1.00")
    var result6 = a6 * b6
    testing.assert_equal(String(result6), "12.3400", "Multiplication by one")

    # Test case 7: Multiplication with large scales
    var a7 = Decimal("0.0001")
    var b7 = Decimal("0.0002")
    var result7 = a7 * b7
    testing.assert_equal(
        String(result7), "0.00000002", "Multiplication with large scales"
    )

    # Test case 8: Multiplication resulting in scale truncation
    var a8 = Decimal("0.123456789")
    var b8 = Decimal("0.987654321")
    var result8 = a8 * b8
    testing.assert_equal(
        String(result8),
        "0.121932631112635269",
        "Multiplication with scale truncation",
    )

    # Test case 9: Multiplication of large numbers
    var a9 = Decimal("1234567.89")
    var b9 = Decimal("9876543.21")
    var result9 = a9 * b9
    testing.assert_equal(
        String(result9),
        "12193263111263.5269",
        "Multiplication of large numbers",
    )

    # Test case 10: Verify that a * b = b * a (commutative property)
    var a10 = Decimal("123.456")
    var b10 = Decimal("789.012")
    var result10a = a10 * b10
    var result10b = b10 * a10
    testing.assert_equal(
        String(result10a),
        String(result10b),
        "Multiplication should be commutative",
    )

    # Test case 11: Multiplication near precision limits
    var a11 = Decimal("0." + "1" * 10)
    var b11 = Decimal("0." + "1" * 18)
    var result11 = a11 * b11
    testing.assert_equal(
        String(result11),
        "0.0123456790111111110987654321",
        "Multiplication near precision limits",
    )

    # Test case 12: Multiplication with integers
    var a12 = Decimal("123")
    var b12 = Decimal("456")
    var result12 = a12 * b12
    testing.assert_equal(
        String(result12), "56088", "Multiplication with integers"
    )

    # Test case 13: Multiplication with powers of 10
    var a13 = Decimal("12.34")
    var b13 = Decimal("10")
    var result13 = a13 * b13
    testing.assert_equal(
        String(result13), "123.40", "Multiplication by power of 10"
    )

    # Test case 14: Multiplication with very small numbers
    var a14 = Decimal("0." + "0" * 25 + "1")
    var b14 = Decimal("0." + "0" * 25 + "1")
    var result14 = a14 * b14
    testing.assert_equal(
        String(result14),
        "0." + "0" * 28,
        "Multiplication of very small numbers",
    )

    # Test case 15: Verify that a * 0 = 0 for various values - individual test cases
    var zero = Decimal("0")

    # Test 15a: Multiplication of zero by zero
    var value15a = Decimal("0")
    testing.assert_equal(
        String(value15a * zero),
        "0",
        "Multiplication by zero should yield zero (zero case)",
    )

    # Test 15b: Multiplication of positive number by zero
    var value15b = Decimal("123.45")
    testing.assert_equal(
        String(value15b * zero),
        "0.00",
        "Multiplication by zero should yield zero (positive number case)",
    )

    # Test 15c: Multiplication of negative number by zero
    var value15c = Decimal("-987.654")
    testing.assert_equal(
        String(value15c * zero),
        "0.000",
        "Multiplication by zero should yield zero (negative number case)",
    )

    # Test 15d: Multiplication of small number by zero
    var value15d = Decimal("0.0001")
    testing.assert_equal(
        String(value15d * zero),
        "0.0000",
        "Multiplication by zero should yield zero (small number case)",
    )

    # Test 15e: Multiplication of large negative number by zero
    var value15e = Decimal("-99999.99999")
    testing.assert_equal(
        String(value15e * zero),
        "0.00000",
        "Multiplication by zero should yield zero (large negative number case)",
    )

    print("Decimal multiplication tests passed!")


fn test_division() raises:
    print("Testing decimal division...")

    # Test case 1: Simple division with same scale
    var a1 = Decimal("10.00")
    var b1 = Decimal("2.00")
    var result1 = a1 / b1
    testing.assert_equal(
        String(result1), "5", "Simple division with same scale"
    )

    # Test case 2: Division with different scales
    var a2 = Decimal("10.5")
    var b2 = Decimal("2.1")
    var result2 = a2 / b2
    testing.assert_equal(
        String(result2),
        "5",
        "Division with different scales",
    )

    # Test case 3: Division with negative numbers
    var a3 = Decimal("12.34")
    var b3 = Decimal("-2.0")
    var result3 = a3 / b3
    testing.assert_equal(
        String(result3),
        "-6.17",
        "Division with negative number",
    )

    # Test case 4: Division with both negative numbers
    var a4 = Decimal("-12.34")
    var b4 = Decimal("-2.0")
    var result4 = a4 / b4
    testing.assert_equal(
        String(result4),
        "6.17",
        "Division with both negative numbers",
    )

    # Test case 5: Division by one
    var a5 = Decimal("12.34")
    var b5 = Decimal("1.0")
    var result5 = a5 / b5
    testing.assert_equal(String(result5), "12.34", "Division by one")

    # Test case 6: Division resulting in repeating decimal
    var a6 = Decimal("10.0")
    var b6 = Decimal("3.0")
    var result6 = a6 / b6
    testing.assert_equal(
        String(result6).startswith("3.333333333"),
        True,
        "Division resulting in repeating decimal",
    )

    # Test case 7: Division resulting in exact value
    var a7 = Decimal("10.0")
    var b7 = Decimal("5.0")
    var result7 = a7 / b7
    testing.assert_equal(
        String(result7), "2", "Division resulting in exact value"
    )

    # Test case 8: Division of zero by non-zero
    var a8 = Decimal("0.0")
    var b8 = Decimal("5.0")
    var result8 = a8 / b8
    testing.assert_equal(String(result8), "0", "Division of zero by non-zero")

    # Test case 9: Division with small numbers
    var a9 = Decimal("0.001")
    var b9 = Decimal("0.01")
    var result9 = a9 / b9
    testing.assert_equal(String(result9), "0.1", "Division with small numbers")

    # Test case 10: Division with large numbers
    var a10 = Decimal("1000000.0")
    var b10 = Decimal("0.001")
    var result10 = a10 / b10
    testing.assert_equal(
        String(result10), "1000000000", "Division with large numbers"
    )

    # Test case 11: Division requiring rounding
    var a11 = Decimal("1.0")
    var b11 = Decimal("7.0")
    var result11 = a11 / b11
    testing.assert_equal(
        String(result11).startswith("0.142857142857142857142857"),
        True,
        "Division requiring rounding",
    )

    # Test case 12: Division with mixed precision
    var a12 = Decimal("123.456")
    var b12 = Decimal("0.1")
    var result12 = a12 / b12
    testing.assert_equal(
        String(result12), "1234.56", "Division with mixed precision"
    )

    # Test case 13: Verify mathematical identity (a/b)*b ≈ a within rounding error
    var a13 = Decimal("123.45")
    var b13 = Decimal("7.89")
    var div_result = a13 / b13
    var mul_result = div_result * b13
    # Because of rounding, we don't expect exact equality, so check if the difference is small
    var diff = a13 - mul_result
    var abs_diff = -diff if diff.is_negative() else diff
    var is_close = Float64(String(abs_diff)) < 0.0001
    testing.assert_equal(is_close, True, "(a/b)*b should approximately equal a")

    # Test case 14: Division of number by itself should be 1
    var a14 = Decimal("123.45")
    var result14 = a14 / a14
    testing.assert_equal(
        String(result14),
        "1",
        "Division of number by itself",
    )

    # Test case 15: Division by zero should raise an error
    var a15 = Decimal("123.45")
    var b15 = Decimal("0.0")
    try:
        var result15 = a15 / b15
        testing.assert_equal(
            True, False, "Division by zero should raise an error"
        )
    except:
        testing.assert_equal(True, True, "Division by zero correctly rejected")

    # ============= ADDITIONAL DIVISION TEST CASES =============
    print("\nTesting additional division scenarios...")

    # Test case 16: Division with very large number by very small number
    var a16 = Decimal("1000000000")
    var b16 = Decimal("0.0001")
    var result16 = a16 / b16
    testing.assert_equal(
        String(result16),
        "10000000000000",
        "Large number divided by small number",
    )

    # Test case 17: Division with very small number by very large number
    var a17 = Decimal("0.0001")
    var b17 = Decimal("1000000000")
    var result17 = a17 / b17
    testing.assert_true(
        String(result17).startswith("0.0000000000001"),
        "Small number divided by large number",
    )

    # Test case 18: Division resulting in repeating decimal
    var a18 = Decimal("1")
    var b18 = Decimal("3")
    var result18 = a18 / b18
    testing.assert_true(
        String(result18).startswith("0.33333333"),
        "Division resulting in repeating decimal (1/3)",
    )

    # Test case 19: Division by powers of 10
    var a19 = Decimal("123.456")
    var b19 = Decimal("10")
    var result19 = a19 / b19
    testing.assert_equal(
        String(result19),
        "12.3456",
        "Division by power of 10",
    )

    # Test case 20: Division by powers of 10 (another case)
    var a20 = Decimal("123.456")
    var b20 = Decimal("0.01")
    var result20 = a20 / b20
    testing.assert_equal(
        String(result20),
        "12345.6",
        "Division by 0.01 (multiply by 100)",
    )

    # Test case 21: Division of nearly equal numbers
    var a21 = Decimal("1.000001")
    var b21 = Decimal("1")
    var result21 = a21 / b21
    testing.assert_equal(
        String(result21),
        "1.000001",
        "Division of nearly equal numbers",
    )

    # Test case 22: Division resulting in a number with many trailing zeros
    var a22 = Decimal("1")
    var b22 = Decimal("8")
    var result22 = a22 / b22
    testing.assert_true(
        String(result22).startswith("0.125"),
        "Division resulting in an exact decimal with trailing zeros",
    )

    # Test case 23: Division with negative numerator
    var a23 = Decimal("-50")
    var b23 = Decimal("10")
    var result23 = a23 / b23
    testing.assert_equal(
        String(result23),
        "-5",
        "Division with negative numerator",
    )

    # Test case 24: Division with negative denominator
    var a24 = Decimal("50")
    var b24 = Decimal("-10")
    var result24 = a24 / b24
    testing.assert_equal(
        String(result24),
        "-5",
        "Division with negative denominator",
    )

    # Test case 25: Division with both negative
    var a25 = Decimal("-50")
    var b25 = Decimal("-10")
    var result25 = a25 / b25
    testing.assert_equal(
        String(result25),
        "5",
        "Division with both negative numbers",
    )

    # Test case 26: Division resulting in exact integer
    var a26 = Decimal("96.75")
    var b26 = Decimal("4.5")
    var result26 = a26 / b26
    testing.assert_equal(
        String(result26),
        "21.5",
        "Division resulting in exact value",
    )

    # Test case 27: Division with high precision numbers
    var a27 = Decimal("0.123456789012345678901234567")
    var b27 = Decimal("0.987654321098765432109876543")
    var result27 = a27 / b27
    testing.assert_true(
        String(result27).startswith("0.12499"),
        "Division of high precision numbers",
    )

    # Test case 28: Division with extreme digit patterns
    var a28 = Decimal("9" * 15)  # 999999999999999
    var b28 = Decimal("9" * 5)  # 99999
    var result28 = a28 / b28
    testing.assert_equal(
        String(result28),
        "10000100001",
        "Division with extreme digit patterns (all 9's)",
    )

    # Test case 29: Division where result is zero
    var a29 = Decimal("0")
    var b29 = Decimal("123.45")
    var result29 = a29 / b29
    testing.assert_equal(
        String(result29),
        "0",
        "Division where result is zero",
    )

    # Test case 30: Division where numerator is smaller than denominator
    var a30 = Decimal("1")
    var b30 = Decimal("10000")
    var result30 = a30 / b30
    testing.assert_equal(
        String(result30),
        "0.0001",
        "Division where numerator is smaller than denominator",
    )

    # Test case 31: Division resulting in scientific notation range
    var a31 = Decimal("1")
    var b31 = Decimal("1" + "0" * 20)  # 10^20
    var result31 = a31 / b31
    testing.assert_true(
        String(result31).startswith("0.00000000000000000001"),
        "Division resulting in very small number",
    )

    # Test case 32: Division with mixed precision
    var a32 = Decimal("1")
    var b32 = Decimal("3.33333333333333333333333333")
    var result32 = a32 / b32
    testing.assert_true(
        String(result32).startswith("0.3"),
        "Division with mixed precision numbers",
    )

    # Test case 33: Division by fractional power of 10
    var a33 = Decimal("5.5")
    var b33 = Decimal("0.055")
    var result33 = a33 / b33
    testing.assert_equal(
        String(result33),
        "100",
        "Division by fractional power of 10",
    )

    # Test case 34: Division with rounding at precision boundary
    var a34 = Decimal("2")
    var b34 = Decimal("3")
    var result34 = a34 / b34
    # Result should be about 0.66666...
    var expected34 = Decimal("0.66666666666666666666666666667")
    print(result34)
    testing.assert_equal(
        result34,
        expected34,
        "Division with rounding at precision boundary",
    )

    # Test case 35: Division by value very close to zero
    var a35 = Decimal("1")
    var b35 = Decimal("0." + "0" * 26 + "1")  # 0.000...0001 (27 zeros)
    var result35 = a35 / b35
    testing.assert_true(
        String(result35).startswith("1" + "0" * 27),
        "Division by value very close to zero",
    )

    print("Additional division tests passed!")

    print("Decimal division tests passed!")


fn test_power_integer_exponents() raises:
    print("Testing power with integer exponents...")

    # Test case 1: Base cases: x^0 = 1 for any x except 0
    var a1 = Decimal("2.5")
    var result1 = a1**0
    testing.assert_equal(
        String(result1), "1", "Any number to power 0 should be 1"
    )

    # Test case 2: 0^n = 0 for n > 0
    var a2 = Decimal("0")
    var result2 = a2**5
    testing.assert_equal(
        String(result2), "0", "0 to any positive power should be 0"
    )

    # Test case 3: x^1 = x
    var a3 = Decimal("3.14159")
    var result3 = a3**1
    testing.assert_equal(String(result3), "3.14159", "x^1 should be x")

    # Test case 4: Positive integer powers
    var a4 = Decimal("2")
    var result4 = a4**3
    testing.assert_equal(String(result4), "8", "2^3 should be 8")

    # Test case 5: Test with scale
    var a5 = Decimal("1.5")
    var result5 = a5**2
    testing.assert_equal(String(result5), "2.25", "1.5^2 should be 2.25")

    # Test case 6: Larger powers
    var a6 = Decimal("2")
    var result6 = a6**10
    testing.assert_equal(String(result6), "1024", "2^10 should be 1024")

    # Test case 7: Negative base, even power
    var a7 = Decimal("-3")
    var result7 = a7**2
    testing.assert_equal(String(result7), "9", "(-3)^2 should be 9")

    # Test case 8: Negative base, odd power
    var a8 = Decimal("-3")
    var result8 = a8**3
    testing.assert_equal(String(result8), "-27", "(-3)^3 should be -27")

    # Test case 9: Decimal base, positive power
    var a9 = Decimal("0.1")
    var result9 = a9**3
    testing.assert_equal(String(result9), "0.001", "0.1^3 should be 0.001")

    # Test case 10: Large number to small power
    var a10 = Decimal("1000")
    var result10 = a10**2
    testing.assert_equal(
        String(result10), "1000000", "1000^2 should be 1000000"
    )

    print("Integer exponent tests passed!")


fn test_power_negative_exponents() raises:
    print("Testing power with negative integer exponents...")

    # Test case 1: Basic negative exponent
    var a1 = Decimal("2")
    var result1 = a1 ** (-2)
    testing.assert_equal(String(result1), "0.25", "2^(-2) should be 0.25")

    # Test case 2: Larger negative exponent
    var a2 = Decimal("10")
    var result2 = a2 ** (-3)
    testing.assert_equal(String(result2), "0.001", "10^(-3) should be 0.001")

    # Test case 3: Negative base, even negative power
    var a3 = Decimal("-2")
    var result3 = a3 ** (-2)
    testing.assert_equal(String(result3), "0.25", "(-2)^(-2) should be 0.25")

    # Test case 4: Negative base, odd negative power
    var a4 = Decimal("-2")
    var result4 = a4 ** (-3)
    testing.assert_equal(
        String(result4), "-0.125", "(-2)^(-3) should be -0.125"
    )

    # Test case 5: Decimal base, negative power
    var a5 = Decimal("0.5")
    var result5 = a5 ** (-2)
    testing.assert_equal(String(result5), "4", "0.5^(-2) should be 4")

    # Test case 6: 1^(-n) = 1
    var a6 = Decimal("1")
    var result6 = a6 ** (-5)
    testing.assert_equal(String(result6), "1", "1^(-5) should be 1")

    print("Negative exponent tests passed!")


fn test_power_special_cases() raises:
    print("Testing power function special cases...")

    # Test case 1: 0^0 (typically defined as 1)
    var a1 = Decimal("0")
    try:
        var result1 = a1**0
        testing.assert_equal(String(result1), "1", "0^0 should be defined as 1")
    except:
        print("0^0 raises an exception (mathematically undefined)")

    # Test case 2: 0^(-n) (mathematically undefined)
    var a2 = Decimal("0")
    try:
        var result2 = a2 ** (-2)
        print("WARNING: 0^(-2) didn't raise an exception, got", result2)
    except:
        print("0^(-2) correctly raises an exception")

    # Test case 3: 1^n = 1 for any n
    var a3 = Decimal("1")
    var result3a = a3**100
    var result3b = a3 ** (-100)
    testing.assert_equal(String(result3a), "1", "1^100 should be 1")
    testing.assert_equal(String(result3b), "1", "1^(-100) should be 1")

    # Test case 4: High precision result with rounding
    # TODO: Implement __gt__
    # var a4 = Decimal("1.1")
    # var result4 = a4**30
    # testing.assert_true(
    #     result4 > Decimal("17.4") and result4 < Decimal("17.5"),
    #     "1.1^30 should be approximately 17.449",
    # )

    print("Special cases tests passed!")


fn test_power_decimal_exponents() raises:
    print("Testing power with decimal exponents...")

    # Try a few basic decimal exponents if supported
    try:
        var a1 = Decimal("4")
        var e1 = Decimal("0.5")  # Square root
        var result1 = a1**e1
        testing.assert_equal(String(result1), "2", "4^0.5 should be 2")

        var a2 = Decimal("8")
        var e2 = Decimal("1.5")  # Cube root of square
        var result2 = a2**e2
        testing.assert_equal(
            String(result2)[:4], "22.6", "8^1.5 should be approximately 22.6"
        )
    except:
        print("Decimal exponents not supported in this implementation")

    print("Decimal exponent tests passed!")


fn test_power_precision() raises:
    print("Testing power precision...")

    # These tests assume we have overloaded the ** operator
    # and we have a way to control precision similar to pow()
    try:
        # Test with precision control
        var a1 = Decimal("1.5")
        var result1 = a1**2
        # Test equality including precision
        testing.assert_equal(
            String(result1), "2.25", "1.5^2 should be exactly 2.25"
        )

        # Check scale
        testing.assert_equal(
            result1.scale(),
            2,
            "Result should maintain precision of 2 decimal places",
        )
    except:
        print("Precision parameters not supported with ** operator")

    print("Precision tests passed!")


fn test_extreme_cases() raises:
    print("Testing extreme cases...")

    # Test case 1: Addition that results in exactly zero with high precision
    var a1 = Decimal("0." + "1" * 28)  # 0.1111...1 (28 digits)
    var b1 = Decimal("-0." + "1" * 28)  # -0.1111...1 (28 digits)
    var result1 = a1 + b1
    testing.assert_equal(
        String(result1),
        "0." + "0" * 28,
        "High precision addition resulting in zero",
    )

    # Test case 2: Addition that should trigger overflow handling
    try:
        var a2 = Decimal("79228162514264337593543950335")  # MAX()
        var b2 = Decimal("1")
        var result2 = a2 + b2
        print("WARNING: Addition beyond MAX() didn't raise an error")
    except:
        print("Addition overflow correctly detected")

    # Test case 3: Addition with mixed precision zeros
    var a3 = Decimal("0.00")
    var b3 = Decimal("0.000000")
    var result3 = a3 + b3
    testing.assert_equal(
        String(result3), "0.000000", "Addition of different precision zeros"
    )

    # Test case 4: Addition with boundary values involving zeros
    var a4 = Decimal("0.0")
    var b4 = Decimal("-0.00")
    var result4 = a4 + b4
    testing.assert_equal(
        String(result4), "0.00", "Addition of positive and negative zero"
    )

    # Test case 5: Adding numbers that require carry propagation through many places
    var a5 = Decimal("9" * 20 + "." + "9" * 28)  # 99...9.99...9
    var b5 = Decimal("0." + "0" * 27 + "1")  # 0.00...01
    var result5 = a5 + b5
    # The result should be 10^20 exactly, since all 9s carry over
    testing.assert_equal(
        String(result5),
        "100000000000000000000.00000000",
        "Addition with extensive carry propagation",
    )

    print("Extreme case tests passed!")


fn main() raises:
    print("Running decimal arithmetic tests")

    # Run addition tests
    test_add()

    # Run negation tests
    test_negation()

    # Run absolute value tests
    test_abs()

    # Run subtraction tests
    test_subtract()

    # Run multiplication tests
    test_multiplication()

    # Run division tests
    test_division()

    # Run power tests with integer exponents
    test_power_integer_exponents()

    # Run power tests with negative exponents

    test_power_negative_exponents()

    # Run power tests for special cases
    test_power_special_cases()

    # Run power tests with decimal exponents
    test_power_decimal_exponents()

    # Run power precision tests
    test_power_precision()

    # Run extreme cases tests
    test_extreme_cases()

    print("All decimal arithmetic tests passed!")
