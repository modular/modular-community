"""Test file for mojo-toml package installation.

This test verifies that mojo-toml is properly installed and can parse TOML.
"""

from toml import parse


fn main() raises:
    """Test basic TOML parsing functionality."""

    print("Testing mojo-toml package installation...")

    # Test 1: Simple key-value parsing
    var config1 = parse('name = "mojo-toml"')
    var name = config1["name"].as_string()
    if name != "mojo-toml":
        raise Error("Test failed: Expected 'mojo-toml', got: " + name)
    print("✓ Test 1 passed: Simple key-value")

    # Test 2: Integer parsing
    var config2 = parse('port = 8080')
    var port = config2["port"].as_int()
    if port != 8080:
        raise Error("Test failed: Expected 8080, got: " + String(port))
    print("✓ Test 2 passed: Integer parsing")

    # Test 3: Array parsing
    var config3 = parse('items = [1, 2, 3]')
    var items = config3["items"].as_array()
    if len(items) != 3:
        raise Error("Test failed: Expected 3 items, got: " + String(len(items)))
    print("✓ Test 3 passed: Array parsing")

    # Test 4: Nested table parsing
    var config4 = parse('[database]\nhost = "localhost"\nport = 5432')
    var db = config4["database"].as_table()
    var host = db["host"].as_string()
    if host != "localhost":
        raise Error("Test failed: Expected 'localhost', got: " + host)
    print("✓ Test 4 passed: Nested tables")

    # Test 5: Dotted keys
    var config5 = parse('a.b.c = "nested"')
    var a_table = config5["a"].as_table()
    var b_table = a_table["b"].as_table()
    var c_value = b_table["c"].as_string()
    if c_value != "nested":
        raise Error("Test failed: Expected 'nested', got: " + c_value)
    print("✓ Test 5 passed: Dotted keys")

    print()
    print("✅ All tests passed! mojo-toml is working correctly.")
    print("Package version: 0.3.0")
