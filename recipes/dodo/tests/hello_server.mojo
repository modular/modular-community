"""A real server built from the installed package. Usage: hello_server PORT"""

from std.sys import argv

from dodo import App, Request, Response, ok


def hello(req: Request) raises -> Response:
    return ok("hello from dodo")


def main() raises:
    var app = App()
    app.get["/"](hello)
    var port = UInt16(Int(argv()[1]))
    app.serve(port=port)
