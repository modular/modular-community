from lightbug_http.cookie import SameSite, Cookie, Duration, Expiration
from small_time.small_time import SmallTime, now
from testing import assert_true, assert_equal
from collections import Optional

fn test_set_cookie() raises:
    cookie = Cookie(
            name="mycookie",
            value="myvalue",
            max_age=Duration(minutes=20),
            expires=Expiration.from_datetime(SmallTime(2037, 1, 22, 12, 0, 10, 0)),
            path=str("/"),
            domain=str("localhost"),
            secure=True,
            http_only=True,
            same_site=SameSite.none,
            partitioned=False
    )
    var header = cookie.to_header()
    var header_value = header.value
    var expected = "mycookie=myvalue; Expires=Thu, 22 Jan 2037 12:00:10 GMT; Max-Age=1200; Domain=localhost; Path=/; Secure; HttpOnly; SameSite=none"
    assert_equal("set-cookie", header.key)
    assert_equal(header_value, expected)


fn test_set_cookie_partial_arguments() raises:
    cookie = Cookie(
            name="mycookie",
            value="myvalue",
            same_site=SameSite.lax
    )
    var header = cookie.to_header()
    var header_value = header.value
    var expected = "mycookie=myvalue; SameSite=lax"
    assert_equal("set-cookie", header.key)
    assert_equal( header_value, expected)


fn test_expires_http_timestamp_format() raises:
    var expected = "Thu, 22 Jan 2037 12:00:10 GMT"
    var http_date = Expiration.from_datetime(SmallTime(2037, 1, 22, 12, 0, 10, 0)).http_date_timestamp()
    assert_true(http_date is not None, msg="Http date is None")
    assert_equal(expected , http_date.value())
