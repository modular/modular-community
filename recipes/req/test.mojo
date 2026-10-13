import req
from std.os import getenv
from std.testing import assert_equal, assert_true


def main() raises:
    var url = getenv("REQ_PACKAGE_TEST_URL")
    var response = req.get(url)
    response.raise_for_status()
    assert_equal(response.status_code, 200)
    assert_equal(response.json()["value"].int_value(), 42)
    assert_true(response.text().byte_length() > 0)

    var client = req.Client(base_url=url + "/", timeout=req.Timeout(5.0))
    var first = client.get("first")
    first.raise_for_status()
    assert_equal(first.json()["value"].int_value(), 42)
    var payload = req.JSONValue.object()
    payload.set("name", req.JSONValue("Mojo"))
    var posted = client.post("post", json=payload)
    posted.raise_for_status()
    assert_equal(posted.json()["name"].string_value(), "Mojo")
    client.close()

    print("Req installed-package HTTP and JSON tests passed")
