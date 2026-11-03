from std.testing import assert_almost_equal

import numojo as nm
from scijo.linalg import det
from scijo.prelude import *


def main() raises:
    var matrix = nm.fromstring[f64]("[[1, 2], [3, 4]]")
    assert_almost_equal(det(matrix), -2.0)
