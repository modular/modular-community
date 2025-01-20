from emberjson import JSON, Null, Array, Object, Value
from emberjson import write_pretty
from testing import *

def main():
    test_json_object()
    test_json_array()
    test_equality()
    test_setter_object()
    test_setter_array()
    test_stringify_array()
    test_pretty_print_array()
    test_pretty_print_object()
    test_trailing_tokens()
    test_bool()
    test_string()
    test_null()
    test_integer()
    test_integer_leading_plus()
    test_integer_negative()
    test_float()
    test_eight_digits_after_dot()
    test_special_case_floats()
    test_float_leading_plus()
    test_float_negative()
    test_float_exponent()
    test_float_exponent_negative()
    test_equality_value()
    test_implicit_conversion()
    test_pretty()
    test_booling()

def test_json_object():
    var s = '{"key": 123}'
    var json = JSON.from_string(s)
    assert_true(json.is_object())
    assert_equal(json.object()["key"].int(), 123)
    assert_equal(json["key"].int(), 123)

    assert_equal(str(json), '{"key":123}')

    assert_equal(len(json), 1)

    with assert_raises():
        _ = json[2]

def test_json_array():
    var s = '[123, 345]'
    var json = JSON.from_string(s)
    assert_true(json.is_array())
    assert_equal(json.array()[0].int(), 123)
    assert_equal(json.array()[1].int(), 345)
    assert_equal(json[0].int(), 123)

    assert_equal(str(json), '[123,345]')

    assert_equal(len(json), 2)

    with assert_raises():
        _ = json["key"]

    json = JSON.from_string("[1, 2, 3]")
    assert_true(json.is_array())
    assert_equal(json[0], 1)
    assert_equal(json[1], 2)
    assert_equal(json[2], 3)

def test_equality():

    var ob = JSON.from_string('{"key": 123}')
    var ob2 = JSON.from_string('{"key": 123}')
    var arr = JSON.from_string('[123, 345]')

    assert_equal(ob, ob2)
    ob["key"] = 456
    assert_not_equal(ob, ob2)
    assert_not_equal(ob, arr)

def test_setter_object():
    var ob: JSON = Object()
    ob["key"] = "foo"
    assert_true("key" in ob)
    assert_equal(ob["key"], "foo")

def test_setter_array():
    var arr: JSON = Array(123, "foo")
    arr[0] = Null()
    assert_true(arr[0].isa[Null]())
    assert_equal(arr[1], "foo")

def test_stringify_array():
    var arr = JSON.from_string('[123,"foo",false,null]')
    assert_equal(str(arr), '[123,"foo",false,null]')

def test_pretty_print_array():
    var arr = JSON.from_string('[123,"foo",false,null]')
    var expected = """[
    123,
    "foo",
    false,
    null
]"""
    assert_equal(expected, write_pretty(arr))

    expected = """[
iamateapot123,
iamateapot"foo",
iamateapotfalse,
iamateapotnull
]"""
    assert_equal(expected, write_pretty(arr, indent=String("iamateapot")))

    arr = JSON.from_string('[123,"foo",false,{"key": null}]')
    expected = """[
    123,
    "foo",
    false,
    {
        "key": null
    }
]"""

    assert_equal(expected, write_pretty(arr))


def test_pretty_print_object():
    var ob = JSON.from_string('{"k1": null, "k2": 123}')
    var expected = """{
    "k1": null,
    "k2": 123
}"""
    assert_equal(expected, write_pretty(ob))

    ob = JSON.from_string('{"key": 123, "k": [123, false, null]}')

    expected = """{
    "key": 123,
    "k": [
        123,
        false,
        null
    ]
}"""

    assert_equal(expected, write_pretty(ob))

    ob = JSON.from_string('{"key": 123, "k": [123, false, [1, 2, 3]]}')
    expected = """{
    "key": 123,
    "k": [
        123,
        false,
        [
            1,
            2,
            3
        ]
    ]
}"""
    assert_equal(expected, write_pretty(ob))


def test_trailing_tokens():
    with assert_raises(contains="Invalid json, expected end of input, recieved: garbage tokens"):
        _ = JSON.from_string('[1, null, false] garbage tokens')

    with assert_raises(contains='Invalid json, expected end of input, recieved: "trailing string"'):
        _ = JSON.from_string('{"key": null} "trailing string"')

def test_bool():
    var s = "false"
    var v = Value.from_string(s)
    assert_true(v.isa[Bool]())
    assert_equal(v.get[Bool](), False)
    assert_equal(str(v), s)

    s = "true"
    v = Value.from_string(s)
    assert_true(v.isa[Bool]())
    assert_equal(v.get[Bool](), True)
    assert_equal(str(v), s)

def test_string():
    var s = '"Some String"'
    var v = Value.from_string(s)
    assert_true(v.isa[String]())
    assert_equal(v.get[String](), "Some String")
    assert_equal(str(v), s)

    s = "\"Escaped\""
    v = Value.from_string(s)
    assert_true(v.isa[String]())
    assert_equal(v.get[String](), "Escaped")
    assert_equal(str(v), s)

def test_null():
    var s = "null"
    var v = Value.from_string(s)
    assert_true(v.isa[Null]())
    assert_equal(v.get[Null](), Null())
    assert_equal(str(v), s)

    with assert_raises(contains="Expected 'null'"):
        _ = Value.from_string("nil")

def test_integer():
    var v = Value.from_string("123")
    assert_true(v.isa[Int]())
    assert_equal(v.get[Int](), 123)
    assert_equal(str(v), "123")

def test_integer_leading_plus():
    v = Value.from_string("+123")
    assert_true(v.isa[Int]())
    assert_equal(v.get[Int](), 123)

def test_integer_negative():
    v = Value.from_string("-123")
    assert_true(v.isa[Int]())
    assert_equal(v.get[Int](), -123)
    assert_equal(str(v), "-123")

def test_float():
    v = Value.from_string("43.5")
    assert_true(v.isa[Float64]())
    assert_almost_equal(v.get[Float64](), 43.5)
    assert_equal(str(v), "43.5")

def test_eight_digits_after_dot():
    v = Value.from_string("342.12345678")
    assert_true(v.isa[Float64]())
    assert_almost_equal(v.get[Float64](), 342.12345678)
    assert_equal(str(v), "342.12345678")

def test_special_case_floats():

    v = Value.from_string('2.2250738585072013e-308')
    assert_almost_equal(v.float(), 2.2250738585072013e-308)
    assert_true(v.isa[Float64]())

    v = Value.from_string('7.2057594037927933e+16')
    assert_true(v.isa[Float64]())
    assert_almost_equal(v.float(), 7.2057594037927933e+16)

    v = Value.from_string('1e000000000000000000001')
    assert_true(v.isa[Float64]())
    assert_almost_equal(v.float(), 1e000000000000000000001)


def test_float_leading_plus():
    v = Value.from_string("+43.5")
    assert_true(v.isa[Float64]())
    assert_almost_equal(v.get[Float64](), 43.5)

def test_float_negative():
    v = Value.from_string("-43.5")
    assert_true(v.isa[Float64]())
    assert_almost_equal(v.get[Float64](), -43.5)

def test_float_exponent():
    v = Value.from_string("43.5e10")
    assert_true(v.isa[Float64]())
    assert_almost_equal(v.get[Float64](), 43.5e10)

def test_float_exponent_negative():
    v = Value.from_string("-43.5e10")
    assert_true(v.isa[Float64]())
    assert_almost_equal(v.get[Float64](), -43.5e10)

def test_equality_value():
    var v1 = Value(34)
    var v2 = Value("Some string")
    var v3 = Value("Some string")
    assert_equal(v2, v3)
    assert_not_equal(v1, v2)


def test_implicit_conversion():
    var val: Value = "a string"
    assert_equal(val.string(), "a string")
    val = 100
    assert_equal(val.int(), 100)
    val = False
    assert_false(val.bool())
    val = 1e10
    assert_almost_equal(val.float(), 1e10)
    val = Null()
    assert_equal(val.null(), Null())
    val = Object()
    assert_equal(val.object(), Object())
    val = Array(1, 2, 3)
    assert_equal(val.array(), Array(1, 2, 3))
    val = JSON()
    assert_equal(val.object(), Object())


def test_pretty():
    var v = Value.from_string("[123, 43564, false]")
    var expected = """[
    123,
    43564,
    false
]"""
    assert_equal(expected, write_pretty(v))

    v = Value.from_string('{"key": 123, "k2": null}')
    expected = """{
    "key": 123,
    "k2": null
}"""

    assert_equal(expected, write_pretty(v))

def test_booling():
    var a: Value = True
    assert_true(a)
    if not a:
        raise Error("Implicit bool failed")


    var trues = Array("some string", 123, 3.43)
    for t in trues:
        assert_true(t[])

    var falsies = Array("", 0, 0.0, False, Null())
    for f in falsies:
        assert_false(f[])