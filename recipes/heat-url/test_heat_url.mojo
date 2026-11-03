from heat_url import VERSION, parse_uri, parse_url, try_parse_url
from std.testing import assert_equal, assert_true


def main() raises:
    assert_equal(String(VERSION), "1.0.0")
    var url = parse_url("https:example.org")
    assert_equal(url.serialize(), "https://example.org/")
    var uri = parse_uri("https:example.org")
    assert_equal(uri.serialize(), "https:example.org")
    assert_true(try_parse_url("https://ex ample.org/") is None)
    print("heat-url package ok")
