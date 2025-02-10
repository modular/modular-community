import numojo as nm
from numojo.prelude import *
from python import Python, PythonObject
from testing.testing import (
    assert_raises,
    assert_equal,
    assert_true,
    assert_almost_equal,
)


fn check[
    dtype: DType
](array: nm.NDArray[dtype], np_sol: PythonObject, st: String) raises:
    var np = Python.import_module("numpy")
    assert_true(np.all(np.equal(array.to_numpy(), np_sol)), st)


fn check_is_close[
    dtype: DType
](array: nm.NDArray[dtype], np_sol: PythonObject, st: String) raises:
    var np = Python.import_module("numpy")
    assert_true(np.all(np.isclose(array.to_numpy(), np_sol, atol=0.1)), st)


fn check_values_close[
    dtype: DType
](value: Scalar[dtype], np_sol: PythonObject, st: String) raises:
    var np = Python.import_module("numpy")
    assert_true(np.isclose(value, np_sol, atol=0.01), st)


def test_arange():
    var np = Python.import_module("numpy")
    check(
        nm.arange[nm.i64](0, 100),
        np.arange(0, 100, dtype=np.int64),
        "Arange is broken",
    )
    check(
        nm.arange[nm.f64](0, 100),
        np.arange(0, 100, dtype=np.float64),
        "Arange is broken",
    )


def test_linspace():
    var np = Python.import_module("numpy")
    check(
        nm.linspace[nm.f64](0, 100),
        np.linspace(0, 100, dtype=np.float64),
        "Linspace is broken",
    )


def main():
    test_arange()
    test_linspace()
