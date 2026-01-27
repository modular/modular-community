"""Test that mojo-yaml package is installed and functional."""

from yaml import parse

fn main() raises:
    # Test basic YAML parsing
    var yaml_str = """
name: test
count: 42
enabled: true
items:
  - first
  - second
config:
  host: localhost
  port: 8080
"""
    
    # Parse YAML - should not crash
    var data = parse(yaml_str)
    
    # Validate basic access
    var name = data.get("name").as_string()
    if name != "test":
        raise Error("Name parsing failed")
    
    var count = data.get("count").as_int()
    if count != 42:
        raise Error("Integer parsing failed")
    
    var enabled = data.get("enabled").as_bool()
    if not enabled:
        raise Error("Boolean parsing failed")
    
    # Test nested mapping
    var config = data.get("config")
    var host = config.get("host").as_string()
    if host != "localhost":
        raise Error("Nested mapping failed")
    
    # Test sequence
    var items = data.get("items")
    if len(items.sequence_value) != 2:
        raise Error("Sequence parsing failed")
    
    print("✓ All tests passed")
