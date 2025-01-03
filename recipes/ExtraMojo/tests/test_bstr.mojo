from ExtraMojo.bstr.bstr import (
    SplitIterator,
    find,
    to_ascii_lowercase,
    to_ascii_uppercase,
)
from memory import Span
from testing import *


fn s(bytes: Span[UInt8]) -> String:
    """Convert bytes to a String."""
    var buffer = String()
    buffer.write_bytes(bytes)
    return buffer


fn test_lowercase_short() raises:
    var example = List("ABCdefgHIjklmnOPQRSTUVWXYZ".as_bytes())
    var answer = "abcdefghijklmnopqrstuvwxyz"
    to_ascii_lowercase(example)
    assert_equal(s(example), s(answer.as_bytes()))


fn test_uppercase_short() raises:
    var example = List("ABCdefgHIjklmnOPQRSTUVWXYZ".as_bytes())
    var answer = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    to_ascii_uppercase(example)
    assert_equal(s(example), s(answer.as_bytes()))


fn test_lowercase() raises:
    var example = List(
        "ABCdefgHIjklmnOPQRSTUVWXYZ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;ABCdefgHIjklmnOPQRSTUVWXYZ"
        .as_bytes()
    )
    var answer = "abcdefghijklmnopqrstuvwxyz;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;abcdefghijklmnopqrstuvwxyz"
    to_ascii_lowercase(example)
    assert_equal(s(example), s(answer.as_bytes()))


fn test_uppercase() raises:
    var example = List(
        "ABCdefgHIjklmnOPQRSTUVWXYZ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;ABCdefgHIjklmnOPQRSTUVWXYZ"
        .as_bytes()
    )
    var answer = "ABCDEFGHIJKLMNOPQRSTUVWXYZ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    to_ascii_uppercase(example)
    assert_equal(s(example), s(answer.as_bytes()))


fn test_lowercase_long() raises:
    var example = List(
        "ABCdefgHIjklmnOPQRSTUVWXYZ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;ABCdefgHIjklmnOPQRSTUVWXYZABCdefgHIjklmnOPQRSTUVWXYZ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;ABCdefgHIjklmnOPQRSTUVWXYZABCdefgHIjklmnOPQRSTUVWXYZ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;ABCdefgHIjklmnOPQRSTUVWXYZABCdefgHIjklmnOPQRSTUVWXYZ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;ABCdefgHIjklmnOPQRSTUVWXYZABCdefgHIjklmnOPQRSTUVWXYZ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;ABCdefgHIjklmnOPQRSTUVWXYZ"
        .as_bytes()
    )
    var answer = "abcdefghijklmnopqrstuvwxyz;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;abcdefghijklmnopqrstuvwxyz"
    to_ascii_lowercase(example)
    assert_equal(s(example), s(answer.as_bytes()))


fn test_uppercase_long() raises:
    var example = List(
        "ABCdefgHIjklmnOPQRSTUVWXYZ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;ABCdefgHIjklmnOPQRSTUVWXYZABCdefgHIjklmnOPQRSTUVWXYZ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;ABCdefgHIjklmnOPQRSTUVWXYZABCdefgHIjklmnOPQRSTUVWXYZ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;ABCdefgHIjklmnOPQRSTUVWXYZABCdefgHIjklmnOPQRSTUVWXYZ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;ABCdefgHIjklmnOPQRSTUVWXYZABCdefgHIjklmnOPQRSTUVWXYZ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;ABCdefgHIjklmnOPQRSTUVWXYZ"
        .as_bytes()
    )
    var answer = "ABCDEFGHIJKLMNOPQRSTUVWXYZ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    to_ascii_uppercase(example)
    assert_equal(s(example), s(answer.as_bytes()))


fn test_find_short() raises:
    var haystack = "ABCDEFGhijklmnop".as_bytes()
    var expected = 4
    var answer = find(haystack, "EFG".as_bytes()).value()
    assert_equal(answer, expected)


fn test_find_medium() raises:
    var haystack = "ABCDEFGhijklmnop0123456789TheKindIguana\nJumpedOver the angry weird fense as it ran away from the seething moon that was swooping down to scoop it up and bring it to outer space.".as_bytes()
    var expected = 171
    var answer = find(haystack, "space".as_bytes()).value()
    assert_equal(answer, expected)


fn test_find_long() raises:
    var haystack = "ABCDEFGhijklmnop0123456789TheKindIguana\nJumpedOver the angry weird fense as it ran away from the seething moon that was swooping down to scoop it up and bring it to outer space.\nThen a really weird thing happened and suddenly 64 moons were swooping down at the Iguana. It tried to turn and tell them it was scalar, but they didn't care all tried to scoop it at once, which resulted in a massive Iguana lock contention.".as_bytes()
    var expected = 373
    var answer = find(haystack, "result".as_bytes()).value()
    assert_equal(answer, expected)


fn test_spilt_iterator() raises:
    var input = "ABCD\tEFGH\tIJKL\nMNOP".as_bytes()
    var expected = List(
        "ABCD".as_bytes(), "EFGH".as_bytes(), "IJKL\nMNOP".as_bytes()
    )
    var output = List[Span[UInt8, StaticConstantOrigin]]()
    for value in SplitIterator(input, ord("\t")):
        output.append(value)
    for i in range(len(expected)):
        assert_equal(s(output[i]), s(expected[i]), "Not equal")


fn test_spilt_iterator_peek() raises:
    var input = "ABCD\tEFGH\tIJKL\nMNOP".as_bytes()
    var expected = List(
        "ABCD".as_bytes(), "EFGH".as_bytes(), "IJKL\nMNOP".as_bytes()
    )
    var iter = SplitIterator(input, ord("\t"))
    var first = iter.__next__()
    var peek = iter.peek()
    var second = iter.__next__()
    assert_equal(s(peek.value()), s(second))
    assert_equal(s(first), s(expected[0]))
    assert_equal(s(second), s(expected[1]))


fn test_spilt_iterator_long() raises:
    var input = "ABCD\tEFGH\tIJKL\nMNOP\tQRST\tUVWXYZ\tABCD\tEFGH\tIJKL\nMNOP\tQRST\tUVWXYZ\tABCD\tEFGH\tIJKL\nMNOP\tQRST\tUVWXYZ\tABCD\tEFGH\tIJKL\nMNOP\tQRST\tUVWXYZ\tABCD\tEFGH\tIJKL\nMNOP\tQRST\tUVWXYZ\tABCD\tEFGH\tIJKL\nMNOP\tQRST\tUVWXYZ\tABCD\tEFGH\tIJKL\nMNOP\tQRST\tUVWXYZ\tABCD\tEFGH\tIJKL\nMNOP\tQRST\tUVWXYZ\tABCD\tEFGH\tIJKL\nMNOP\tQRST\tUVWXYZ\tABCD\tEFGH\tIJKL\nMNOP\tQRST\tUVWXYZ\tABCD\tEFGH\tIJKL\nMNOP\tQRST\tUVWXYZ\tABCD\tEFGH\tIJKL\nMNOP\tQRST\tUVWXYZ\tABCD\tEFGH\tIJKL\nMNOP\tQRST\tUVWXYZ\tABCD\tEFGH\tIJKL\nMNOP\tQRST\tUVWXYZ\tABCD\tEFGH\tIJKL\nMNOP\tQRST\tUVWXYZ\tABCD\tEFGH\tIJKL\nMNOP\tQRST\tUVWXYZ\tABCD\tEFGH\tIJKL\nMNOP\tQRST\tUVWXYZ\tABCD\tEFGH\tIJKL\nMNOP\tQRST\tUVWXYZ".as_bytes()
    var expected = List(
        "ABCD".as_bytes(),
        "EFGH".as_bytes(),
        "IJKL\nMNOP".as_bytes(),
        "QRST".as_bytes(),
        "UVWXYZ".as_bytes(),
        "ABCD".as_bytes(),
        "EFGH".as_bytes(),
        "IJKL\nMNOP".as_bytes(),
        "QRST".as_bytes(),
        "UVWXYZ".as_bytes(),
        "ABCD".as_bytes(),
        "EFGH".as_bytes(),
        "IJKL\nMNOP".as_bytes(),
        "QRST".as_bytes(),
        "UVWXYZ".as_bytes(),
        "ABCD".as_bytes(),
        "EFGH".as_bytes(),
        "IJKL\nMNOP".as_bytes(),
        "QRST".as_bytes(),
        "UVWXYZ".as_bytes(),
        "ABCD".as_bytes(),
        "EFGH".as_bytes(),
        "IJKL\nMNOP".as_bytes(),
        "QRST".as_bytes(),
        "UVWXYZ".as_bytes(),
        "ABCD".as_bytes(),
        "EFGH".as_bytes(),
        "IJKL\nMNOP".as_bytes(),
        "QRST".as_bytes(),
        "UVWXYZ".as_bytes(),
        "ABCD".as_bytes(),
        "EFGH".as_bytes(),
        "IJKL\nMNOP".as_bytes(),
        "QRST".as_bytes(),
        "UVWXYZ".as_bytes(),
        "ABCD".as_bytes(),
        "EFGH".as_bytes(),
        "IJKL\nMNOP".as_bytes(),
        "QRST".as_bytes(),
        "UVWXYZ".as_bytes(),
        "ABCD".as_bytes(),
        "EFGH".as_bytes(),
        "IJKL\nMNOP".as_bytes(),
        "QRST".as_bytes(),
        "UVWXYZ".as_bytes(),
        "ABCD".as_bytes(),
        "EFGH".as_bytes(),
        "IJKL\nMNOP".as_bytes(),
        "QRST".as_bytes(),
        "UVWXYZ".as_bytes(),
        "ABCD".as_bytes(),
        "EFGH".as_bytes(),
        "IJKL\nMNOP".as_bytes(),
        "QRST".as_bytes(),
        "UVWXYZ".as_bytes(),
        "ABCD".as_bytes(),
        "EFGH".as_bytes(),
        "IJKL\nMNOP".as_bytes(),
        "QRST".as_bytes(),
        "UVWXYZ".as_bytes(),
        "ABCD".as_bytes(),
        "EFGH".as_bytes(),
        "IJKL\nMNOP".as_bytes(),
        "QRST".as_bytes(),
        "UVWXYZ".as_bytes(),
        "ABCD".as_bytes(),
        "EFGH".as_bytes(),
        "IJKL\nMNOP".as_bytes(),
        "QRST".as_bytes(),
        "UVWXYZ".as_bytes(),
        "ABCD".as_bytes(),
        "EFGH".as_bytes(),
        "IJKL\nMNOP".as_bytes(),
        "QRST".as_bytes(),
        "UVWXYZ".as_bytes(),
        "ABCD".as_bytes(),
        "EFGH".as_bytes(),
        "IJKL\nMNOP".as_bytes(),
        "QRST".as_bytes(),
        "UVWXYZ".as_bytes(),
        "ABCD".as_bytes(),
        "EFGH".as_bytes(),
        "IJKL\nMNOP".as_bytes(),
        "QRST".as_bytes(),
        "UVWXYZ".as_bytes(),
        "ABCD".as_bytes(),
        "EFGH".as_bytes(),
        "IJKL\nMNOP".as_bytes(),
        "QRST".as_bytes(),
        "UVWXYZ".as_bytes(),
    )
    var output = List[Span[UInt8, StaticConstantOrigin]]()
    for value in SplitIterator(input, ord("\t")):
        output.append(value)
    for i in range(len(expected)):
        assert_equal(s(output[i]), s(expected[i]), "Not equal")
