"""Checks an installed dodo package: the App and every native library.

Run by the conda recipe's test step against the installed package, not the
source tree.
"""

from std.testing import assert_equal, assert_true

from dodo import App, JsonBody, PathParams, Request
from dodo.crypto.digest import sha256
from dodo.db.sqlite import Sqlite
from dodo.http import Method
from dodo.http.encoding import (
    compress_brotli,
    compress_gzip,
    decode_content,
)
from dodo.json.ffi.simdjson import simdjson_available


@fieldwise_init
struct NamePath(PathParams):
    var name: String


@fieldwise_init
struct NewUser(JsonBody):
    var name: String
    var age: Int


@fieldwise_init
struct Greeting(Movable):
    var message: String


@fieldwise_init
struct Created(Movable):
    var name: String
    var age: Int


def hello(req: Request, path: NamePath) raises -> Greeting:
    return Greeting(String("hi ", path.name))


def create(req: Request, body: NewUser) raises -> Created:
    return Created(body.name, body.age)


def check_app() raises:
    var app = App()
    app.get["/hello/{name}"](hello)
    app.post["/users"](create)

    var get_req = Request(method=Method.GET, url="/hello/dodo")
    var got = app.serve(get_req)
    assert_equal(got.status, 200)
    assert_equal(got.text(), '{"message":"hi dodo"}')

    var body = String('{"name":"ada","age":36}')
    var post_req = Request(
        method=Method.POST, url="/users", body=List[UInt8](body.as_bytes())
    )
    var posted = app.serve(post_req)
    assert_equal(posted.status, 200)
    assert_equal(posted.text(), '{"name":"ada","age":36}')


def check_compression() raises:
    var text = String("dodo dodo dodo dodo dodo dodo")
    var gzipped = compress_gzip(text.as_bytes())
    var unzipped = decode_content(gzipped, "gzip")
    assert_equal(String(unsafe_from_utf8=unzipped), text)

    var brotli = compress_brotli(text.as_bytes())
    var unbrotli = decode_content(brotli, "br")
    assert_equal(String(unsafe_from_utf8=unbrotli), text)


def check_openssl() raises:
    var empty = String("")
    var digest = sha256(empty.as_bytes())
    assert_equal(len(digest), 32)
    assert_equal(Int(digest[0]), 0xE3)


def check_simdjson() raises:
    assert_true(simdjson_available(), "libdodo_simdjson.so did not load")


def check_sqlite() raises:
    var db = Sqlite("sqlite://:memory:")
    assert_equal(db.query("select 41 + 1 as n")[0].int("n"), 42)


def main() raises:
    check_app()
    check_compression()
    check_openssl()
    check_simdjson()
    check_sqlite()
    print("dodo package smoke test passed")
