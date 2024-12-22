import numojo as nm
from numojo.prelude import *
from numojo import mat
from python import Python, PythonObject
from testing.testing import assert_raises, assert_equal, assert_true, assert_almost_equal


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


def test_logspace():
    var np = Python.import_module("numpy")
    check_is_close(
        nm.logspace[nm.f64](0, 100, 5),
        np.logspace(0, 100, 5, dtype=np.float64),
        "Logspace is broken",
    )


def test_geomspace():
    var np = Python.import_module("numpy")
    check_is_close(
        nm.geomspace[nm.f64](1, 100, 5),
        np.geomspace(1, 100, 5, dtype=np.float64),
        "Logspace is broken",
    )


def test_zeros():
    var np = Python.import_module("numpy")
    check(
        nm.zeros[f64](Shape(10, 10, 10, 10)),
        np.zeros((10, 10, 10, 10), dtype=np.float64),
        "Zeros is broken",
    )


def test_ones_from_shape():
    var np = Python.import_module("numpy")
    check(
        nm.ones[nm.f64](Shape(10, 10, 10, 10)),
        np.ones((10, 10, 10, 10), dtype=np.float64),
        "Ones is broken",
    )


def test_ones_from_list():
    var np = Python.import_module("numpy")
    check(
        nm.ones[nm.f64](List[Int](10, 10, 10, 10)),
        np.ones((10, 10, 10, 10), dtype=np.float64),
        "Ones is broken",
    )


def test_ones_from_vlist():
    var np = Python.import_module("numpy")
    check(
        nm.ones[nm.f64](VariadicList[Int](10, 10, 10, 10)),
        np.ones((10, 10, 10, 10), dtype=np.float64),
        "Ones is broken",
    )


def test_full():
    var np = Python.import_module("numpy")
    check(
        nm.full[nm.f64](Shape(10, 10, 10, 10), fill_value=10),
        np.full((10, 10, 10, 10), 10, dtype=np.float64),
        "Full is broken",
    )


def test_full_from_shape():
    var np = Python.import_module("numpy")
    check(
        nm.full[nm.f64](Shape(10, 10, 10, 10), fill_value=10),
        np.full((10, 10, 10, 10), 10, dtype=np.float64),
        "Full is broken",
    )


def test_full_from_list():
    var np = Python.import_module("numpy")
    check(
        nm.full[nm.f64](List[Int](10, 10, 10, 10), fill_value=10),
        np.full((10, 10, 10, 10), 10, dtype=np.float64),
        "Full is broken",
    )


def test_full_from_vlist():
    var np = Python.import_module("numpy")
    check(
        nm.full[nm.f64](VariadicList[Int](10, 10, 10, 10), fill_value=10),
        np.full((10, 10, 10, 10), 10, dtype=np.float64),
        "Full is broken",
    )


def test_identity():
    var np = Python.import_module("numpy")
    check(
        nm.identity[nm.i64](100),
        np.identity(100, dtype=np.int64),
        "Identity is broken",
    )


def test_eye():
    var np = Python.import_module("numpy")
    check(
        nm.eye[nm.i64](100, 100),
        np.eye(100, 100, dtype=np.int64),
        "Eye is broken",
    )


def test_fromstring():
    var A = nm.fromstring("[[[1,2],[3,4]],[[5,6],[7,8]]]")
    var B = nm.array[DType.int32](String("[0.1, -2.3, 41.5, 19.29145, -199]"))
    print(A)
    print(B)


def test_fromstring_complicated():
    var s = """
    [[[[1,2,10],
       [3,4,2]],
       [[5,6,4],
       [7,8,10]]],
     [[[1,2,12],
       [3,4,41]],
       [[5,6,12],
       [7,8,99]]]]
    """
    var A = nm.fromstring(s)
    print(A)


def test_diag():
    var np = Python.import_module("numpy")
    var x_nm = nm.arange[f32](0, 9, step=1)
    x_nm.reshape(3, 3)
    var x_np = np.arange(0, 9, step=1).reshape(3, 3)

    x_nm_k0 = nm.diag[f32](x_nm, k=0)
    x_np_k0 = np.diag(x_np, k=0)
    check(x_nm_k0, x_np_k0, "Diag is broken (k=0)")

    x_nm_k1 = nm.diag[f32](x_nm, k=1)
    x_np_k1 = np.diag(x_np, k=1)
    check(x_nm_k1, x_np_k1, "Diag is broken (k=1)")

    x_nm_km1 = nm.diag[f32](x_nm, k=-1)
    x_np_km1 = np.diag(x_np, k=-1)
    check(x_nm_km1, x_np_km1, "Diag is broken (k=-1)")

    x_nm_rev_k1 = nm.diag[f32](x_nm_k0, k=0)
    x_np_rev_k1 = np.diag(x_np_k0, k=0)
    check(x_nm_rev_k1, x_np_rev_k1, "Diag reverse is broken (k=0)")

    x_nm_rev_km1 = nm.diag[f32](x_nm_km1, k=-1)
    x_np_rev_km1 = np.diag(x_np_km1, k=-1)
    check(x_nm_rev_km1, x_np_rev_km1, "Diag reverse is broken (k=-1)")


def test_diagflat():
    var np = Python.import_module("numpy")
    var nm_arr = nm.arange[nm.i64](0, 9, 1)
    nm_arr.reshape(3, 3)
    var np_arr = np.arange(0, 9, 1).reshape(3, 3)

    var x_nm = nm.diagflat[nm.i64](nm_arr, k=0)
    var x_np = np.diagflat(np_arr, k=0)
    check(x_nm, x_np, "Diagflat is broken (k=0)")

    var x_nm_k1 = nm.diagflat[nm.i64](nm_arr, k=1)
    var x_np_k1 = np.diagflat(np_arr, k=1)
    check(x_nm_k1, x_np_k1, "Diagflat is broken (k=1)")

    var x_nm_km1 = nm.diagflat[nm.i64](nm_arr, k=-1)
    var x_np_km1 = np.diagflat(np_arr, k=-1)
    check(x_nm_km1, x_np_km1, "Diagflat is broken (k=-1)")


def test_tri():
    var np = Python.import_module("numpy")

    var x_nm = nm.tri[nm.f32](3, 4, k=0)
    var x_np = np.tri(3, 4, k=0, dtype=np.float32)
    check(x_nm, x_np, "Tri is broken (k=0)")

    var x_nm_k1 = nm.tri[nm.f32](3, 4, k=1)
    var x_np_k1 = np.tri(3, 4, k=1, dtype=np.float32)
    check(x_nm_k1, x_np_k1, "Tri is broken (k=1)")

    var x_nm_km1 = nm.tri[nm.f32](3, 4, k=-1)
    var x_np_km1 = np.tri(3, 4, k=-1, dtype=np.float32)
    check(x_nm_km1, x_np_km1, "Tri is broken (k=-1)")


def test_tril():
    var np = Python.import_module("numpy")
    var nm_arr = nm.arange[nm.f32](0, 9, 1)
    nm_arr.reshape(3, 3)
    var np_arr = np.arange(0, 9, 1, dtype=np.float32).reshape(3, 3)

    var x_nm = nm.tril[nm.f32](nm_arr, k=0)
    var x_np = np.tril(np_arr, k=0)
    check(x_nm, x_np, "Tril is broken (k=0)")

    var x_nm_k1 = nm.tril[nm.f32](nm_arr, k=1)
    var x_np_k1 = np.tril(np_arr, k=1)
    check(x_nm_k1, x_np_k1, "Tril is broken (k=1)")

    var x_nm_km1 = nm.tril[nm.f32](nm_arr, k=-1)
    var x_np_km1 = np.tril(np_arr, k=-1)
    check(x_nm_km1, x_np_km1, "Tril is broken (k=-1)")

    # Test with higher dimensional array
    var nm_arr_3d = nm.arange[nm.f32](0, 60, 1)
    nm_arr_3d.reshape(3, 4, 5)
    var np_arr_3d = np.arange(0, 60, 1, dtype=np.float32).reshape(3, 4, 5)

    var x_nm_3d = nm.tril[nm.f32](nm_arr_3d, k=0)
    var x_np_3d = np.tril(np_arr_3d, k=0)
    check(x_nm_3d, x_np_3d, "Tril is broken for 3D array (k=0)")

    var x_nm_3d_k1 = nm.tril[nm.f32](nm_arr_3d, k=1)
    var x_np_3d_k1 = np.tril(np_arr_3d, k=1)
    check(x_nm_3d_k1, x_np_3d_k1, "Tril is broken for 3D array (k=1)")

    var x_nm_3d_km1 = nm.tril[nm.f32](nm_arr_3d, k=-1)
    var x_np_3d_km1 = np.tril(np_arr_3d, k=-1)
    check(x_nm_3d_km1, x_np_3d_km1, "Tril is broken for 3D array (k=-1)")


def test_triu():
    var np = Python.import_module("numpy")
    var nm_arr = nm.arange[nm.f32](0, 9, 1)
    nm_arr.reshape(3, 3)
    var np_arr = np.arange(0, 9, 1, dtype=np.float32).reshape(3, 3)

    var x_nm = nm.triu[nm.f32](nm_arr, k=0)
    var x_np = np.triu(np_arr, k=0)
    check(x_nm, x_np, "Triu is broken (k=0)")

    var x_nm_k1 = nm.triu[nm.f32](nm_arr, k=1)
    var x_np_k1 = np.triu(np_arr, k=1)
    check(x_nm_k1, x_np_k1, "Triu is broken (k=1)")

    var x_nm_km1 = nm.triu[nm.f32](nm_arr, k=-1)
    var x_np_km1 = np.triu(np_arr, k=-1)
    check(x_nm_km1, x_np_km1, "Triu is broken (k=-1)")

    # Test with higher dimensional array
    var nm_arr_3d = nm.arange[nm.f32](0, 60, 1)
    nm_arr_3d.reshape(3, 4, 5)
    var np_arr_3d = np.arange(0, 60, 1, dtype=np.float32).reshape(3, 4, 5)

    var x_nm_3d = nm.triu[nm.f32](nm_arr_3d, k=0)
    var x_np_3d = np.triu(np_arr_3d, k=0)
    check(x_nm_3d, x_np_3d, "Triu is broken for 3D array (k=0)")

    var x_nm_3d_k1 = nm.triu[nm.f32](nm_arr_3d, k=1)
    var x_np_3d_k1 = np.triu(np_arr_3d, k=1)
    check(x_nm_3d_k1, x_np_3d_k1, "Triu is broken for 3D array (k=1)")

    var x_nm_3d_km1 = nm.triu[nm.f32](nm_arr_3d, k=-1)
    var x_np_3d_km1 = np.triu(np_arr_3d, k=-1)
    check(x_nm_3d_km1, x_np_3d_km1, "Tril is broken for 3D array (k=-1)")


def test_vander():
    var np = Python.import_module("numpy")
    var nm_arr = nm.arange[nm.f32](1, 5, 1)
    var np_arr = np.arange(1, 5, 1, dtype=np.float32)

    var x_nm = nm.vander[nm.f32](nm_arr)
    var x_np = np.vander(np_arr)
    check(x_nm, x_np, "Vander is broken (default)")

    var x_nm_N3 = nm.vander[nm.f32](nm_arr, N=3)
    var x_np_N3 = np.vander(np_arr, N=3)
    check(x_nm_N3, x_np_N3, "Vander is broken (N=3)")

    var x_nm_inc = nm.vander[nm.f32](nm_arr, increasing=True)
    var x_np_inc = np.vander(np_arr, increasing=True)
    check(x_nm_inc, x_np_inc, "Vander is broken (increasing=True)")

    var x_nm_N3_inc = nm.vander[nm.f32](nm_arr, N=3, increasing=True)
    var x_np_N3_inc = np.vander(np_arr, N=3, increasing=True)
    check(x_nm_N3_inc, x_np_N3_inc, "Vander is broken (N=3, increasing=True)")

    # Test with different dtype
    var nm_arr_int = nm.arange[nm.i32](1, 5, 1)
    var np_arr_int = np.arange(1, 5, 1, dtype=np.int32)

    var x_nm_int = nm.vander[nm.i32](nm_arr_int)
    var x_np_int = np.vander(np_arr_int)
    check(x_nm_int, x_np_int, "Vander is broken (int32)")


def test_arr_manipulation():
    var np = Python.import_module("numpy")

    # Test arange
    var A = nm.arange[nm.i16](1, 7, 1)
    var np_A = np.arange(1, 7, 1, dtype=np.int16)
    check_is_close(A, np_A, "Arange operation")

    # Test flip
    var flipped_A = nm.flip(A)
    var np_flipped_A = np.flip(np_A)
    check_is_close(flipped_A, np_flipped_A, "Flip operation")

    # Test reshape
    A.reshape(2, 3)
    np_A = np_A.reshape((2, 3))
    check_is_close(A, np_A, "Reshape operation")

    # # Test ravel
    # var B = nm.arange[nm.i16](0, 12, 1)
    # var np_B = np.arange(0, 12, 1, dtype=np.int16)
    # B.reshape(3, 2, 2, order="F")
    # np_B = np_B.reshape((3, 2, 2), order='F')
    # var raveled_B = nm.ravel(B, order="C")
    # var np_raveled_B = np.ravel(np_B, order='C')
    # check_is_close(raveled_B, np_raveled_B, "Ravel operation")


def test_transpose():
    var np = Python.import_module("numpy")
    var A = nm.random.randn(2)
    var Anp = A.to_numpy()
    check_is_close(
        nm.transpose(A), np.transpose(Anp), "1-d `transpose` is broken."
    )
    A = nm.random.randn(2, 3)
    Anp = A.to_numpy()
    check_is_close(
        nm.transpose(A), np.transpose(Anp), "2-d `transpose` is broken."
    )
    A = nm.random.randn(2, 3, 4)
    Anp = A.to_numpy()
    check_is_close(
        nm.transpose(A), np.transpose(Anp), "3-d `transpose` is broken."
    )
    A = nm.random.randn(2, 3, 4, 5)
    Anp = A.to_numpy()
    check_is_close(
        nm.transpose(A), np.transpose(Anp), "4-d `transpose` is broken."
    )
    check_is_close(
        A.T(), np.transpose(Anp), "4-d `transpose` with `.T` is broken."
    )
    check_is_close(
        nm.transpose(A, axes=List(1, 3, 0, 2)),
        np.transpose(Anp, [1, 3, 0, 2]),
        "4-d `transpose` with arbitrary `axes` is broken.",
    )


def test_setitem():
    var np = Python.import_module("numpy")
    var arr = nm.NDArray(Shape(4, 4))
    var np_arr = arr.to_numpy()
    arr.itemset(List(2, 2), 1000)
    np_arr[(2, 2)] = 1000
    check_is_close(arr, np_arr, "Itemset is broken")

def test_bool_masks_gt():
    var np = Python.import_module("numpy")
    var np_A = np.arange(0, 24, dtype=np.int16).reshape((3, 2, 4))
    var A = nm.arange[nm.i16](0, 24)
    A.reshape(3, 2, 4)

    var np_gt = np_A > 10
    var gt = A > Scalar[nm.i16](10)
    check(gt, np_gt, "Greater than mask")

    # Test greater than or equal
    var np_ge = np_A >= 10
    var ge = A >= Scalar[nm.i16](10)
    check(ge, np_ge, "Greater than or equal mask")


def test_bool_masks_lt():
    var np = Python.import_module("numpy")

    # Create NumPy and NuMojo arrays using arange and reshape
    var np_A = np.arange(0, 24, dtype=np.int16).reshape((3, 2, 4))
    var A = nm.arange[nm.i16](0, 24)
    A.reshape(3, 2, 4)

    # Test less than
    var np_lt = np_A < 10
    var lt = A < Scalar[nm.i16](10)
    check(lt, np_lt, "Less than mask")

    # Test less than or equal
    var np_le = np_A <= 10
    var le = A <= Scalar[nm.i16](10)
    check(le, np_le, "Less than or equal mask")


def test_bool_masks_eq():
    var np = Python.import_module("numpy")

    # Create NumPy and NuMojo arrays using arange and reshape
    var np_A = np.arange(0, 24, dtype=np.int16).reshape((3, 2, 4))
    var A = nm.arange[nm.i16](0, 24)
    A.reshape(3, 2, 4)

    # Test equal
    # var np_eq = np_A == 10 # has a python interop problem.
    var np_eq = np.equal(np_A, 10)
    var eq = A == Scalar[nm.i16](10)
    check(eq, np_eq, "Equal mask")

    # Test not equal
    # var np_ne = np_A != 10 # has a python interop problem.
    var np_ne = np.not_equal(np_A, 10)
    var ne = A != Scalar[nm.i16](10)
    check(ne, np_ne, "Not equal mask")

    # Test masked array
    var np_mask = np_A[np_A > 10]
    var mask = A[A > Scalar[nm.i16](10)]
    check(mask, np_mask, "Masked array")

def test_constructors():
    # Test NDArray constructor with different input types
    var arr1 = NDArray[f32](Shape(3, 4, 5))
    assert_true(arr1.ndim == 3, "NDArray constructor: ndim")
    assert_true(arr1.shape[0] == 3, "NDArray constructor: shape element 0")
    assert_true(arr1.shape[1] == 4, "NDArray constructor: shape element 1")
    assert_true(arr1.shape[2] == 5, "NDArray constructor: shape element 2")
    assert_true(arr1.size == 60, "NDArray constructor: size")
    assert_true(arr1.dtype == DType.float32, "NDArray constructor: dtype")

    var arr2 = NDArray[f32](VariadicList[Int](3, 4, 5))
    assert_true(
        arr2.shape[0] == 3,
        "NDArray constructor with VariadicList: shape element 0",
    )
    assert_true(
        arr2.shape[1] == 4,
        "NDArray constructor with VariadicList: shape element 1",
    )
    assert_true(
        arr2.shape[2] == 5,
        "NDArray constructor with VariadicList: shape element 2",
    )

    var arr3 = nm.full[f32](Shape(3, 4, 5), fill_value=Scalar[f32](10.0))
    # maybe it's better to return a scalar for arr[0, 0, 0]
    assert_equal(
        arr3[idx(0, 0, 0)], 10.0, "NDArray constructor with fill value"
    )

    var arr4 = NDArray[f32](List[Int](3, 4, 5))
    assert_true(
        arr4.shape[0] == 3, "NDArray constructor with List: shape element 0"
    )
    assert_true(
        arr4.shape[1] == 4, "NDArray constructor with List: shape element 1"
    )
    assert_true(
        arr4.shape[2] == 5, "NDArray constructor with List: shape element 2"
    )

    var arr5 = NDArray[f32](NDArrayShape(3, 4, 5))
    assert_true(
        arr5.shape[0] == 3,
        "NDArray constructor with NDArrayShape: shape element 0",
    )
    assert_true(
        arr5.shape[1] == 4,
        "NDArray constructor with NDArrayShape: shape element 1",
    )
    assert_true(
        arr5.shape[2] == 5,
        "NDArray constructor with NDArrayShape: shape element 2",
    )

    var arr6 = nm.array[f32](
        data=List[SIMD[f32, 1]](1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        shape=List[Int](2, 5),
    )
    assert_true(
        arr6.shape[0] == 2,
        "NDArray constructor with data and shape: shape element 0",
    )
    assert_true(
        arr6.shape[1] == 5,
        "NDArray constructor with data and shape: shape element 1",
    )
    assert_equal(
        arr6[idx(1, 4)],
        10.0,
        "NDArray constructor with data: value check",
    )

# ===-----------------------------------------------------------------------===#
# Sums, products, differences
# ===-----------------------------------------------------------------------===#


def test_sum():
    var np = Python.import_module("numpy")
    var A = nm.random.randn(6, 6, 6)
    var Anp = A.to_numpy()

    check_values_close(
        nm.sum(A),
        np.sum(Anp),
        String("`sum` fails. {} vs {}."),
    )
    for i in range(3):
        check_is_close(
            nm.sum(A, axis=i),
            np.sum(Anp, axis=i),
            String("`sum` by axis {} fails.".format(i)),
        )


def test_add_array():
    var np = Python.import_module("numpy")
    var arr = nm.arange[nm.f64](0, 15)

    check(nm.add[nm.f64](arr, 5.0), np.arange(0, 15) + 5, "Add array + scalar")
    check(
        nm.add[nm.f64](arr, arr),
        np.arange(0, 15) + np.arange(0, 15),
        "Add array + array",
    )


def test_add_array_par():
    var np = Python.import_module("numpy")
    var arr = nm.arange[nm.f64](0, 500)

    check(
        nm.add[nm.f64, backend = nm.core._math_funcs.Vectorized](arr, 5.0),
        np.arange(0, 500) + 5,
        "Add array + scalar",
    )
    check(
        nm.add[nm.f64, backend = nm.core._math_funcs.Vectorized](arr, arr),
        np.arange(0, 500) + np.arange(0, 500),
        "Add array + array",
    )


def test_sin():
    var np = Python.import_module("numpy")
    var arr = nm.arange[nm.f64](0, 15)

    check_is_close(
        nm.sin[nm.f64](arr), np.sin(np.arange(0, 15)), "Add array + scalar"
    )


def test_sin_par():
    var np = Python.import_module("numpy")
    var arr = nm.arange[nm.f64](0, 15)

    check_is_close(
        nm.sin[
            nm.f64,
            backend = nm.core._math_funcs.Vectorized,
        ](arr),
        np.sin(np.arange(0, 15)),
        "Add array + scalar",
    )


# ! MATMUL RESULTS IN A SEGMENTATION FAULT EXCEPT FOR NAIVE ONE, BUT NAIVE OUTPUTS WRONG VALUES


def test_matmul_small():
    var np = Python.import_module("numpy")
    var arr = nm.ones[i8](Shape(4, 4))
    var np_arr = np.ones((4, 4), dtype=np.int8)
    check_is_close(
        arr @ arr, np.matmul(np_arr, np_arr), "Dunder matmul is broken"
    )


def test_matmul():
    var np = Python.import_module("numpy")
    var arr = nm.arange[nm.f64](0, 100)
    arr.reshape(10, 10)
    var np_arr = np.arange(0, 100).reshape(10, 10)
    check_is_close(
        arr @ arr, np.matmul(np_arr, np_arr), "Dunder matmul is broken"
    )
    # The only matmul that currently works is par (__matmul__)
    # check_is_close(nm.matmul_tiled_unrolled_parallelized(arr,arr),np.matmul(np_arr,np_arr),"TUP matmul is broken")


def test_matmul_1dx2d():
    var np = Python.import_module("numpy")
    var arr1 = nm.random.randn(4)
    var arr2 = nm.random.randn(4, 8)
    var nparr1 = arr1.to_numpy()
    var nparr2 = arr2.to_numpy()
    check_is_close(
        arr1 @ arr2, np.matmul(nparr1, nparr2), "Dunder matmul is broken"
    )


def test_matmul_2dx1d():
    var np = Python.import_module("numpy")
    var arr1 = nm.random.randn(11, 4)
    var arr2 = nm.random.randn(4)
    var nparr1 = arr1.to_numpy()
    var nparr2 = arr2.to_numpy()
    check_is_close(
        arr1 @ arr2, np.matmul(nparr1, nparr2), "Dunder matmul is broken"
    )


def test_inv():
    var np = Python.import_module("numpy")
    var arr = nm.core.random.rand(100, 100)
    var np_arr = arr.to_numpy()
    check_is_close(
        nm.math.linalg.inv(arr), np.linalg.inv(np_arr), "Inverse is broken"
    )


def test_solve():
    var np = Python.import_module("numpy")
    var A = nm.core.random.randn(100, 100)
    var B = nm.core.random.randn(100, 50)
    var A_np = A.to_numpy()
    var B_np = B.to_numpy()
    check_is_close(
        nm.linalg.solve(A, B),
        np.linalg.solve(A_np, B_np),
        "Solve is broken",
    )


# ===-----------------------------------------------------------------------===#
# Main functions
# ===-----------------------------------------------------------------------===#


fn check_matrices_equal[
    dtype: DType
](matrix: mat.Matrix[dtype], np_sol: PythonObject, st: String) raises:
    var np = Python.import_module("numpy")
    assert_true(np.all(np.equal(np.matrix(matrix.to_numpy()), np_sol)), st)


fn check_matrices_close[
    dtype: DType
](matrix: mat.Matrix[dtype], np_sol: PythonObject, st: String) raises:
    var np = Python.import_module("numpy")
    assert_true(
        np.all(np.isclose(np.matrix(matrix.to_numpy()), np_sol, atol=0.01)), st
    )


fn check_values_close[
    dtype: DType
](value: Scalar[dtype], np_sol: PythonObject, st: String) raises:
    var np = Python.import_module("numpy")
    assert_true(np.isclose(value, np_sol, atol=0.01), st)


# ===-----------------------------------------------------------------------===#
# Manipulation
# ===-----------------------------------------------------------------------===#


def test_manipulation():
    var np = Python.import_module("numpy")
    var A = mat.rand[f64]((10, 10)) * 1000
    var Anp = np.matrix(A.to_numpy())
    check_matrices_equal(
        A.astype[nm.i32](),
        Anp.astype(np.int32),
        "`astype` is broken",
    )

    check_matrices_equal(
        A.reshape((50, 2)),
        Anp.reshape((50, 2)),
        "Reshape is broken",
    )

    _ = A.resize((1000, 100))
    _ = Anp.resize((1000, 100))
    check_matrices_equal(
        A,
        Anp,
        "Resize is broken",
    )

# ===-----------------------------------------------------------------------===#
# Arithmetic
# ===-----------------------------------------------------------------------===#


def test_arithmetic():
    var np = Python.import_module("numpy")
    var A = mat.rand[f64]((10, 10))
    var B = mat.rand[f64]((10, 10))
    var C = mat.rand[f64]((10, 1))
    var Ap = A.to_numpy()
    var Bp = B.to_numpy()
    var Cp = C.to_numpy()
    check_matrices_close(A + B, Ap + Bp, "Add is broken")
    check_matrices_close(A - B, Ap - Bp, "Sub is broken")
    check_matrices_close(A * B, Ap * Bp, "Mul is broken")
    check_matrices_close(A @ B, np.matmul(Ap, Bp), "Matmul is broken")
    check_matrices_close(A + C, Ap + Cp, "Add (broadcast) is broken")
    check_matrices_close(A - C, Ap - Cp, "Sub (broadcast) is broken")
    check_matrices_close(A * C, Ap * Cp, "Mul (broadcast) is broken")
    check_matrices_close(A / C, Ap / Cp, "Div (broadcast) is broken")
    check_matrices_close(A + 1, Ap + 1, "Add (to int) is broken")
    check_matrices_close(A - 1, Ap - 1, "Sub (to int) is broken")
    check_matrices_close(A * 1, Ap * 1, "Mul (to int) is broken")
    check_matrices_close(A / 1, Ap / 1, "Div (to int) is broken")
    check_matrices_close(A**2, np.power(Ap, 2), "Pow (to int) is broken")
    check_matrices_close(A**0.5, np.power(Ap, 0.5), "Pow (to int) is broken")


def test_logic():
    var np = Python.import_module("numpy")
    var A = mat.ones((5, 1))
    var B = mat.ones((5, 1))
    var L = mat.fromstring[i8](
        "[[0,0,0],[0,0,1],[1,1,1],[1,0,0]]", shape=(4, 3)
    )
    var Anp = np.matrix(A.to_numpy())
    var Bnp = np.matrix(B.to_numpy())
    var Lnp = np.matrix(L.to_numpy())

    check_matrices_equal(A > B, Anp > Bnp, "gt is broken")
    check_matrices_equal(A < B, Anp < Bnp, "lt is broken")
    assert_true(
        np.equal(mat.all(L), np.all(Lnp)),
        "`all` is broken",
    )
    for i in range(2):
        check_matrices_close(
            mat.all(L, axis=i),
            np.all(Lnp, axis=i),
            String("`all` by axis {i} is broken"),
        )
    assert_true(
        np.equal(mat.any(L), np.any(Lnp)),
        "`any` is broken",
    )
    for i in range(2):
        check_matrices_close(
            mat.any(L, axis=i),
            np.any(Lnp, axis=i),
            String("`any` by axis {i} is broken"),
        )


# ===-----------------------------------------------------------------------===#
# Linear algebra
# ===-----------------------------------------------------------------------===#


def test_linalg():
    var np = Python.import_module("numpy")
    var A = mat.rand[f64]((100, 100))
    var B = mat.rand[f64]((100, 100))
    var E = mat.fromstring("[[1,2,3],[4,5,6],[7,8,9],[10,11,12]]", shape=(4, 3))
    var Y = mat.rand((100, 1))
    var Anp = A.to_numpy()
    var Bnp = B.to_numpy()
    var Ynp = Y.to_numpy()
    var Enp = E.to_numpy()
    check_matrices_close(
        mat.solve(A, B),
        np.linalg.solve(Anp, Bnp),
        "Solve is broken",
    )
    check_matrices_close(
        mat.inv(A),
        np.linalg.inv(Anp),
        "Inverse is broken",
    )
    check_matrices_close(
        mat.lstsq(A, Y),
        np.linalg.lstsq(Anp, Ynp)[0],
        "Least square is broken",
    )
    check_matrices_close(
        A.transpose(),
        Anp.transpose(),
        "Transpose is broken",
    )
    check_matrices_close(
        Y.transpose(),
        Ynp.transpose(),
        "Transpose is broken",
    )
    assert_true(
        np.all(np.isclose(mat.det(A), np.linalg.det(Anp), atol=0.1)),
        "Determinant is broken",
    )
    for i in range(-10, 10):
        assert_true(
            np.all(
                np.isclose(
                    mat.trace(E, offset=i), np.trace(Enp, offset=i), atol=0.1
                )
            ),
            "Trace is broken",
        )


# ===-----------------------------------------------------------------------===#
# Mathematics
# ===-----------------------------------------------------------------------===#


def test_math():
    var np = Python.import_module("numpy")
    var A = mat.rand[f64]((100, 100))
    var Anp = np.matrix(A.to_numpy())

    assert_true(
        np.all(np.isclose(mat.sum(A), np.sum(Anp), atol=0.1)),
        "`sum` is broken",
    )
    for i in range(2):
        check_matrices_close(
            mat.sum(A, axis=i),
            np.sum(Anp, axis=i),
            String("`sum` by axis {i} is broken"),
        )

    assert_true(
        np.all(np.isclose(mat.prod(A), np.prod(Anp), atol=0.1)),
        "`prod` is broken",
    )
    for i in range(2):
        check_matrices_close(
            mat.prod(A, axis=i),
            np.prod(Anp, axis=i),
            String("`prod` by axis {i} is broken"),
        )

    check_matrices_close(
        mat.cumsum(A),
        np.cumsum(Anp),
        "`cumsum` is broken",
    )
    for i in range(2):
        check_matrices_close(
            mat.cumsum(A, axis=i),
            np.cumsum(Anp, axis=i),
            String("`cumsum` by axis {i} is broken"),
        )

    check_matrices_close(
        mat.cumprod(A),
        np.cumprod(Anp),
        "`cumprod` is broken",
    )
    for i in range(2):
        check_matrices_close(
            mat.cumprod(A, axis=i),
            np.cumprod(Anp, axis=i),
            String("`cumprod` by axis {i} is broken"),
        )


def test_trigonometric():
    var np = Python.import_module("numpy")
    var A = mat.rand[f64]((100, 100))
    var Anp = np.matrix(A.to_numpy())
    check_matrices_close(mat.sin(A), np.sin(Anp), "sin is broken")
    check_matrices_close(mat.cos(A), np.cos(Anp), "cos is broken")
    check_matrices_close(mat.tan(A), np.tan(Anp), "tan is broken")
    check_matrices_close(mat.arcsin(A), np.arcsin(Anp), "arcsin is broken")
    check_matrices_close(mat.asin(A), np.arcsin(Anp), "asin is broken")
    check_matrices_close(mat.arccos(A), np.arccos(Anp), "arccos is broken")
    check_matrices_close(mat.acos(A), np.arccos(Anp), "acos is broken")
    check_matrices_close(mat.arctan(A), np.arctan(Anp), "arctan is broken")
    check_matrices_close(mat.atan(A), np.arctan(Anp), "atan is broken")


def test_hyperbolic():
    var np = Python.import_module("numpy")
    var A = mat.fromstring("[[1,2,3],[4,5,6],[7,8,9]]", shape=(3, 3))
    var B = A / 10
    var Anp = np.matrix(A.to_numpy())
    var Bnp = np.matrix(B.to_numpy())
    check_matrices_close(mat.sinh(A), np.sinh(Anp), "sinh is broken")
    check_matrices_close(mat.cosh(A), np.cosh(Anp), "cosh is broken")
    check_matrices_close(mat.tanh(A), np.tanh(Anp), "tanh is broken")
    check_matrices_close(mat.arcsinh(A), np.arcsinh(Anp), "arcsinh is broken")
    check_matrices_close(mat.asinh(A), np.arcsinh(Anp), "asinh is broken")
    check_matrices_close(mat.arccosh(A), np.arccosh(Anp), "arccosh is broken")
    check_matrices_close(mat.acosh(A), np.arccosh(Anp), "acosh is broken")
    check_matrices_close(mat.arctanh(B), np.arctanh(Bnp), "arctanh is broken")
    check_matrices_close(mat.atanh(B), np.arctanh(Bnp), "atanh is broken")


# ===-----------------------------------------------------------------------===#
# Statistics
# ===-----------------------------------------------------------------------===#


def test_statistics():
    var np = Python.import_module("numpy")
    var A = mat.rand[f64]((100, 100))
    var Anp = np.matrix(A.to_numpy())

    assert_true(
        np.all(np.isclose(mat.mean(A), np.mean(Anp), atol=0.1)),
        "`mean` is broken",
    )
    for i in range(2):
        check_matrices_close(
            mat.mean(A, i),
            np.mean(Anp, i),
            String("`mean` is broken for {i}-dimension"),
        )

    assert_true(
        np.all(np.isclose(mat.variance(A), np.`var`(Anp), atol=0.1)),
        "`variance` is broken",
    )
    for i in range(2):
        check_matrices_close(
            mat.variance(A, i),
            np.`var`(Anp, i),
            String("`variance` is broken for {i}-dimension"),
        )

    assert_true(
        np.all(np.isclose(mat.std(A), np.std(Anp), atol=0.1)),
        "`std` is broken",
    )
    for i in range(2):
        check_matrices_close(
            mat.std(A, i),
            np.std(Anp, i),
            String("`std` is broken for {i}-dimension"),
        )


def test_sorting():
    var np = Python.import_module("numpy")
    var A = mat.rand[f64]((10, 10))
    var Anp = np.matrix(A.to_numpy())

    check_matrices_close(
        mat.sort(A), np.sort(Anp, axis=None), String("Sort is broken")
    )
    for i in range(2):
        check_matrices_close(
            mat.sort(A, axis=i),
            np.sort(Anp, axis=i),
            String("Sort by axis {} is broken").format(i),
        )

    check_matrices_close(
        mat.argsort(A),
        np.argsort(Anp, axis=None),
        String("Argsort is broken")
        + str(mat.argsort(A))
        + str(np.argsort(Anp, axis=None)),
    )
    for i in range(2):
        check_matrices_close(
            mat.argsort(A, axis=i),
            np.argsort(Anp, axis=i),
            String("Argsort by axis {} is broken").format(i),
        )


def test_searching():
    var np = Python.import_module("numpy")
    var A = mat.rand[f64]((10, 10))
    var Anp = np.matrix(A.to_numpy())

    check_values_close(
        mat.max(A), np.max(Anp, axis=None), String("`max` is broken")
    )
    for i in range(2):
        check_matrices_close(
            mat.max(A, axis=i),
            np.max(Anp, axis=i),
            String("`max` by axis {} is broken").format(i),
        )

    check_values_close(
        mat.argmax(A), np.argmax(Anp, axis=None), String("`argmax` is broken")
    )
    for i in range(2):
        check_matrices_close(
            mat.argmax(A, axis=i),
            np.argmax(Anp, axis=i),
            String("`argmax` by axis {} is broken").format(i),
        )

    check_values_close(
        mat.min(A), np.min(Anp, axis=None), String("`min` is broken.")
    )
    for i in range(2):
        check_matrices_close(
            mat.min(A, axis=i),
            np.min(Anp, axis=i),
            String("`min` by axis {} is broken").format(i),
        )

    check_values_close(
        mat.argmin(A), np.argmin(Anp, axis=None), String("`argmin` is broken.")
    )
    for i in range(2):
        check_matrices_close(
            mat.argmin(A, axis=i),
            np.argmin(Anp, axis=i),
            String("`argmin` by axis {} is broken").format(i),
        )


def test_rand():
    """Test random array generation with specified shape."""
    var arr = nm.random.rand[nm.f64](3, 5, 2)
    assert_true(arr.shape[0] == 3, "Shape of random array")
    assert_true(arr.shape[1] == 5, "Shape of random array")
    assert_true(arr.shape[2] == 2, "Shape of random array")


def test_randminmax():
    """Test random array generation with min and max values."""
    var arr_variadic = nm.random.rand[nm.f64](10, 10, 10, min=1, max=2)
    var arr_list = nm.random.rand[nm.f64](List[Int](10, 10, 10), min=3, max=4)
    var arr_variadic_mean = nm.cummean(arr_variadic)
    var arr_list_mean = nm.cummean(arr_list)
    assert_almost_equal(
        arr_variadic_mean,
        1.5,
        msg="Mean of random array within min and max",
        atol=0.1,
    )
    assert_almost_equal(
        arr_list_mean,
        3.5,
        msg="Mean of random array within min and max",
        atol=0.1,
    )


def test_randn():
    """Test random array generation with normal distribution."""
    var arr_variadic_01 = nm.random.randn[nm.f64](
        20, 20, 20, mean=0.0, variance=1.0
    )
    var arr_variadic_31 = nm.random.randn[nm.f64](
        20, 20, 20, mean=3.0, variance=1.0
    )
    var arr_variadic_12 = nm.random.randn[nm.f64](
        20, 20, 20, mean=1.0, variance=3.0
    )

    var arr_variadic_mean01 = nm.cummean(arr_variadic_01)
    var arr_variadic_mean31 = nm.cummean(arr_variadic_31)
    var arr_variadic_mean12 = nm.cummean(arr_variadic_12)
    var arr_variadic_var01 = nm.cumvariance(arr_variadic_01)
    var arr_variadic_var31 = nm.cumvariance(arr_variadic_31)
    var arr_variadic_var12 = nm.cumvariance(arr_variadic_12)

    assert_almost_equal(
        arr_variadic_mean01,
        0,
        msg="Mean of random array with mean 0 and variance 1",
        atol=0.1,
    )
    assert_almost_equal(
        arr_variadic_mean31,
        3,
        msg="Mean of random array with mean 3 and variance 1",
        atol=0.1,
    )
    assert_almost_equal(
        arr_variadic_mean12,
        1,
        msg="Mean of random array with mean 1 and variance 2",
        atol=0.1,
    )

    assert_almost_equal(
        arr_variadic_var01,
        1,
        msg="Variance of random array with mean 0 and variance 1",
        atol=0.1,
    )
    assert_almost_equal(
        arr_variadic_var31,
        1,
        msg="Variance of random array with mean 3 and variance 1",
        atol=0.1,
    )
    assert_almost_equal(
        arr_variadic_var12,
        3,
        msg="Variance of random array with mean 1 and variance 2",
        atol=0.1,
    )


def test_randn_list():
    """Test random array generation with normal distribution."""
    var arr_list_01 = nm.random.randn[nm.f64](
        List[Int](20, 20, 20), mean=0.0, variance=1.0
    )
    var arr_list_31 = nm.random.randn[nm.f64](
        List[Int](20, 20, 20), mean=3.0, variance=1.0
    )
    var arr_list_12 = nm.random.randn[nm.f64](
        List[Int](20, 20, 20), mean=1.0, variance=2.0
    )

    var arr_list_mean01 = nm.cummean(arr_list_01)
    var arr_list_mean31 = nm.cummean(arr_list_31)
    var arr_list_mean12 = nm.cummean(arr_list_12)
    var arr_list_var01 = nm.cumvariance(arr_list_01)
    var arr_list_var31 = nm.cumvariance(arr_list_31)
    var arr_list_var12 = nm.cumvariance(arr_list_12)

    assert_almost_equal(
        arr_list_mean01,
        0,
        msg="Mean of random array with mean 0 and variance 1",
        atol=0.1,
    )
    assert_almost_equal(
        arr_list_mean31,
        3,
        msg="Mean of random array with mean 3 and variance 1",
        atol=0.1,
    )
    assert_almost_equal(
        arr_list_mean12,
        1,
        msg="Mean of random array with mean 1 and variance 2",
        atol=0.1,
    )

    assert_almost_equal(
        arr_list_var01,
        1,
        msg="Variance of random array with mean 0 and variance 1",
        atol=0.1,
    )
    assert_almost_equal(
        arr_list_var31,
        1,
        msg="Variance of random array with mean 3 and variance 1",
        atol=0.1,
    )
    assert_almost_equal(
        arr_list_var12,
        2,
        msg="Variance of random array with mean 1 and variance 2",
        atol=0.1,
    )


def test_rand_exponential():
    """Test random array generation with exponential distribution."""
    var arr_variadic = nm.random.rand_exponential[nm.f64](20, 20, 20, rate=2.0)
    var arr_list = nm.random.rand_exponential[nm.f64](
        List[Int](20, 20, 20), rate=0.5
    )

    var arr_variadic_mean = nm.cummean(arr_variadic)
    var arr_list_mean = nm.cummean(arr_list)

    # For exponential distribution, mean = 1 / rate
    assert_almost_equal(
        arr_variadic_mean,
        1 / 2,
        msg="Mean of exponential distribution with rate 2.0",
        atol=0.1,
    )
    assert_almost_equal(
        arr_list_mean,
        1 / 0.5,
        msg="Mean of exponential distribution with rate 0.5",
        atol=0.2,
    )

    # For exponential distribution, variance = 1 / (rate^2)
    var arr_variadic_var = nm.cumvariance(arr_variadic)
    var arr_list_var = nm.cumvariance(arr_list)

    assert_almost_equal(
        arr_variadic_var,
        1 / (2),
        msg="Variance of exponential distribution with rate 2.0",
        atol=0.1,
    )
    assert_almost_equal(
        arr_list_var,
        1 / (0.5),
        msg="Variance of exponential distribution with rate 0.5",
        atol=0.5,
    )

    # Test that all values are non-negative
    for i in range(arr_variadic.num_elements()):
        assert_true(
            arr_variadic._buf[i] >= 0,
            "Exponential distribution should only produce non-negative values",
        )

    for i in range(arr_list.num_elements()):
        assert_true(
            arr_list._buf[i] >= 0,
            "Exponential distribution should only produce non-negative values",
        )

def test_convolve2d():
    var sp = Python.import_module("scipy")
    in1 = nm.random.rand(6, 6)
    in2 = nm.fromstring("[[1, 0], [0, -1]]")

    npin1 = in1.to_numpy()
    npin2 = in2.to_numpy()

    res1 = nm.science.signal.convolve2d(in1, in2)
    res2 = sp.signal.convolve2d(npin1, npin2, mode="valid")
    check(
        res1,
        res2,
        "test_convolve2d failed #2\n" + str(res1) + "\n" + str(res2),
    )

def test_slicing_getter1():
    var np = Python.import_module("numpy")

    # Test C-order array slicing
    nm_arr = nm.arange[nm.f32](0.0, 24.0, step=1)
    nm_arr.reshape(2, 3, 4, order="C")
    np_arr = np.arange(0, 24, dtype=np.float32).reshape(2, 3, 4)

    # Test case 1: Slicing all dimensions
    nm_slice1 = nm_arr[:, :, 1:2]
    np_sliced1 = np.take(
        np.take(
            np.take(np_arr, np.arange(0, 2), axis=0), np.arange(0, 3), axis=1
        ),
        np.arange(1, 2),
        axis=2,
    )
    np_sliced1 = np.squeeze(np_sliced1, axis=2)
    check(nm_slice1, np_sliced1, "3D array slicing (C-order) [:, :, 1:2]")


def test_slicing_getter2():
    var np = Python.import_module("numpy")

    # Test C-order array slicing
    nm_arr = nm.arange[nm.f32](0.0, 24.0, step=1)
    nm_arr.reshape(2, 3, 4, order="C")
    np_arr = np.arange(0, 24, dtype=np.float32).reshape(2, 3, 4)

    # Test case 2: Slicing with start and end indices
    nm_slice2 = nm_arr[0:1, 1:3, 2:4]
    np_sliced2 = np.take(
        np.take(
            np.take(np_arr, np.arange(0, 1), axis=0), np.arange(1, 3), axis=1
        ),
        np.arange(2, 4),
        axis=2,
    )
    check(nm_slice2, np_sliced2, "3D array slicing (C-order) [0:1, 1:3, 2:4]")


def test_slicing_getter3():
    var np = Python.import_module("numpy")

    # Test C-order array slicing
    nm_arr = nm.arange[nm.f32](0.0, 24.0, step=1)
    nm_arr.reshape(2, 3, 4, order="C")
    np_arr = np.arange(0, 24, dtype=np.float32).reshape(2, 3, 4)

    # Test case 3: Slicing with mixed start, end, and step values
    nm_slice3 = nm_arr[1:, 0:2, ::2]
    np_sliced3 = np.take(
        np.take(
            np.take(np_arr, np.arange(1, np_arr.shape[0]), axis=0),
            np.arange(0, 2),
            axis=1,
        ),
        np.arange(0, np_arr.shape[2], 2),
        axis=2,
    )
    check(nm_slice3, np_sliced3, "3D array slicing (C-order) [1:, 0:2, ::2]")


def test_slicing_getter4():
    var np = Python.import_module("numpy")

    # Test C-order array slicing
    nm_arr = nm.arange[nm.f32](0.0, 24.0, step=1)
    nm_arr.reshape(2, 3, 4, order="C")
    np_arr = np.arange(0, 24, dtype=np.float32).reshape(2, 3, 4)

    # Test case 4: Slicing with step
    nm_slice4 = nm_arr[::2, ::2, ::2]
    np_sliced4 = np.take(
        np.take(
            np.take(np_arr, np.arange(0, np_arr.shape[0], 2), axis=0),
            np.arange(0, np_arr.shape[1], 2),
            axis=1,
        ),
        np.arange(0, np_arr.shape[2], 2),
        axis=2,
    )
    check(nm_slice4, np_sliced4, "3D array slicing (C-order) [::2, ::2, ::2]")


def test_slicing_getter5():
    var np = Python.import_module("numpy")

    # Test C-order array slicing
    nm_arr = nm.arange[nm.f32](0.0, 24.0, step=1)
    nm_arr.reshape(2, 3, 4, order="C")
    np_arr = np.arange(0, 24, dtype=np.float32).reshape(2, 3, 4)

    # Test case 5: Slicing with combination of integer and slices
    nm_slice5 = nm_arr[1:2, :, 1:3]
    np_sliced5 = np.take(
        np.take(np_arr[1], np.arange(0, np_arr.shape[1]), axis=0),
        np.arange(1, 3),
        axis=1,
    )
    check(nm_slice5, np_sliced5, "3D array slicing (C-order) [1, :, 1:3]")

def test_sort_1d():
    var arr = nm.core.random.rand[nm.i16](25, min=0, max=100)
    var np = Python.import_module("numpy")
    arr.sort()
    np_arr_sorted = np.sort(arr.to_numpy())
    return check[nm.i16](arr, np_arr_sorted, "quick sort is broken")
