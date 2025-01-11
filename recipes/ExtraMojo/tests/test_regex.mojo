from ExtraMojo.regex.simple_re import *
from testing import *


fn main() raises:
    test_start_anchor()
    test_end_anchor()
    test_dot()
    test_star()
    test_literal()
    test_dot_star()
    test_all()
    print("SUCCESS")


fn test_start_anchor() raises:
    var re = "^cat"
    assert_true(is_match(re, "cats of a feather"))
    assert_false(is_match(re, "bird cats of a cat"))


fn test_end_anchor() raises:
    var re = "what$"
    assert_true(is_match(re, "It is what"))
    assert_false(is_match(re, "what is in the box"))


fn test_dot() raises:
    var re = "w.t"
    assert_true(is_match(re, "Is that a witty remark?"))
    assert_false(is_match(re, "wt is that what thing there"))


fn test_star() raises:
    var re = "wha*"
    assert_true(is_match(re, "what am I doing here"))
    assert_true(is_match(re, "whaaaaaaat am I doing here"))
    assert_true(is_match(re, "wht am I doing here"))
    assert_false(is_match(re, "wt am I doing here"))


fn test_literal() raises:
    var re = "ACTG"
    assert_true(is_match(re, "TGGGACTGCCCACTG"))
    assert_true(is_match(re, "CTGGGACGCCCACTG"))
    assert_false(is_match(re, "CTGGGACGCCCACG"))


fn test_dot_star() raises:
    var re = "STAR.*"
    assert_true(is_match(re, "STAR"))
    assert_true(is_match(re, "I'M A STAR"))
    assert_true(is_match(re, "I'M A STARXXXXXXX"))
    assert_true(is_match(re, "I'M A STARS"))
    assert_true(is_match(re, "I'M A STAR!!!!!"))
    assert_false(is_match(re, "I'm not a STArsss"))


fn test_all() raises:
    assert_true(is_match("^cat.*$", "catsssssss"))
    assert_false(is_match("^cat.*$", "many catsssssss"))
