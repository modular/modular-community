from python import Python
from tensor import Tensor, rand
from testing import assert_true, assert_equal

from bridge.numpy import ndarray_to_tensor, tensor_to_ndarray


def test_tensor_identity_transformation():
    """Test that `ndarray_to_tensor` is inverse of `tensor_to_ndarray`."""
    var values = List[Float64](1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    var in_matrix = Tensor[DType.float64](shape=(3, 2), list=values)
    var np_array = tensor_to_ndarray(in_matrix)
    var out_matrix = ndarray_to_tensor[DType.float64](np_array)
    assert_equal(in_matrix, out_matrix)


def test_numpy_identity_transformation():
    """Test that `tensor_to_ndarray` is inverse of `ndarray_to_tensor`."""
    var np = Python.import_module("numpy")
    var in_array = np.arange(6).reshape(3, 2)
    var tensor = ndarray_to_tensor[DType.float64](in_array)
    var out_array = tensor_to_ndarray(tensor)
    assert_equal(in_array, out_array)
