"""Public-package smoke fixture: one ordinary passing TestSuite file."""

from std.testing import assert_equal, TestSuite


def test_public_package_executes() raises:
    assert_equal(2 + 2, 4)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
