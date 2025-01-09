from testing import assert_true, assert_false, assert_equal, assert_not_equal
from infrared.algebra import *


def main():
    test_eq()
    test_ne()
    test_subspace_constructor()
    test_getattr()
    test_normalized()
    test_add()
    test_sub()
    test_mul()
    test_sandwich()


def test_eq():
    alias g3 = Signature(3, 0, 0)
    assert_true(Multivector[g3, g3.empty_mask()]().__eq__(Float64(0)))
    assert_true(Multivector[g3, g3.vector_mask()](1, 2, 3).__eq__(Multivector[g3, g3.vector_mask()](1, 2, 3)))
    assert_false(Multivector[g3, g3.vector_mask()](1, 2, 3).__eq__(Multivector[g3, g3.vector_mask()](1, 4, 3)))
    assert_false(Multivector[g3, g3.vector_mask()](1, 2, 3).__eq__(Float64(1)))


def test_ne():
    alias g3 = Signature(3, 0, 0)
    assert_false(Multivector[g3, g3.empty_mask()]().__ne__(Float64(0)))
    assert_false(Multivector[g3, g3.vector_mask()](1, 2, 3).__ne__(Multivector[g3, g3.vector_mask()](1, 2, 3)))
    assert_true(Multivector[g3, g3.vector_mask()](1, 2, 3).__ne__(Multivector[g3, g3.vector_mask()](1, 4, 3)))
    assert_true(Multivector[g3, g3.vector_mask()](1, 2, 3).__ne__(Float64(1)))


def test_subspace_constructor():
    assert_true(scalar[G3](6) == Multivector[G3](6))
    assert_true(scalar[G3](0) != Multivector[G3](6))

    assert_true(vector[G3](1, 2, 3) == Multivector[G3, G3.vector_mask()](1, 2, 3))
    assert_true(vector[G3](1, 0, 3) != Multivector[G3, G3.vector_mask()](1, 2, 3))

    assert_true(bivector[G3](4, 5, 6) == Multivector[G3, G3.bivector_mask()](4, 5, 6))
    assert_true(bivector[G3](4, 0, 6) != Multivector[G3, G3.bivector_mask()](4, 5, 6))


def test_getattr():
    alias g3 = Signature(3, 0, 0)
    alias scalar_vector_mask = List(True, True, True, True, False, False, False, False)
    assert_equal(Multivector[g3](6).s, Float64(6))
    assert_equal(Multivector[g3, g3.vector_mask()](7, 8, 9).s, Float64(0))
    assert_equal(Multivector[g3, scalar_vector_mask](6, 7, 8, 9).s, Float64(6))


def test_normalized():
    alias g3 = Signature(3, 0, 0)
    assert_true(Multivector[g3, g3.vector_mask()](1, 2, 3).normalized() == Multivector[g3, g3.vector_mask()](0.2672612419124244, 0.53452248382484879, 0.80178372573727319))
    assert_true(Multivector[g3, g3.bivector_mask()](1, 2, 3).normalized() == Multivector[g3, g3.bivector_mask()](0.2672612419124244, 0.53452248382484879, 0.80178372573727319))


def test_add():
    alias g3 = Signature(3, 0, 0)
    alias scalar_vector_mask = List(True, True, True, True, False, False, False, False)
    assert_true(Multivector[g3, g3.vector_mask()](1, 2, 3).__add__(Multivector[g3, g3.vector_mask()](1, 2, 3)) == Multivector[g3, g3.vector_mask()](2, 4, 6))
    assert_true(Multivector[g3, g3.vector_mask()](1, 2, 3).__add__(Float64(1)) == Multivector[g3, scalar_vector_mask](1, 1, 2, 3))


def test_sub():
    alias g3 = Signature(3, 0, 0)
    alias scalar_vector_mask = List(True, True, True, True, False, False, False, False)
    assert_true(Multivector[g3, g3.vector_mask()](2, 4, 6).__sub__(Multivector[g3, g3.vector_mask()](1, 2, 3)) == Multivector[g3, g3.vector_mask()](1, 2, 3))
    assert_true(Multivector[g3, g3.vector_mask()](1, 2, 3).__sub__(Float64(1)) == Multivector[g3, scalar_vector_mask](-1, 1, 2, 3))


def test_mul():
    alias g3 = Signature(3, 0, 0)
    assert_true(Multivector[g3, g3.vector_mask()](1, 2, 3).__mul__(Float64(2)) == Multivector[g3, g3.vector_mask()](2, 4, 6))
    assert_true(Multivector[g3, g3.vector_mask()](1, 2, 3).__mul__(Multivector[g3, g3.antiscalar_mask()](1)) == Multivector[g3, g3.bivector_mask()](3, -2, 1))

    alias ug3 = Signature(1, 1, 1, flip_ze = False)
    alias v1_mask = List(False, True, False, False, False, False, False, False)
    assert_true(Multivector[ug3, v1_mask](2).__mul__(Multivector[ug3, v1_mask](2)) == Float64(4))
    alias v2_mask = List(False, False, True, False, False, False, False, False)
    assert_true(Multivector[ug3, v2_mask](2).__mul__(Multivector[ug3, v2_mask](2)) == Float64(-4))
    alias v3_mask = List(False, False, False, True, False, False, False, False)
    assert_true(Multivector[ug3, v3_mask](2).__mul__(Multivector[ug3, v3_mask](2)) == Float64(0))

    alias pg3 = Signature(3, 0, 1, flip_ze = True)
    alias niltrivector_mask = List(False, False, False, False, False, False, False, False, False, False, False, True, True, True, False, False)
    assert_true(Multivector[pg3, pg3.vector_mask()](1, 2, 3, 4).__mul__(Float64(2)) == Multivector[pg3, pg3.vector_mask()](2, 4, 6, 8))
    assert_true(Multivector[pg3, pg3.vector_mask()](1, 2, 3, 4).__mul__(Multivector[pg3, pg3.antiscalar_mask()](1)) == Multivector[pg3, niltrivector_mask](-4, 3, -2))


def test_sandwich():
    alias g3 = Signature(3, 0, 0)
    assert_true(Multivector[g3, g3.even_mask()](0, 1, 0, 0)(Multivector[g3, g3.vector_mask()](1, 2, 3)) == Multivector[g3, g3.vector_mask()](-1, -2, 3))