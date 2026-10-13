import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def respond(self, body):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.respond(b'{"value":42}')

    def do_POST(self):
        self.respond(self.rfile.read(int(self.headers["Content-Length"])))

    def log_message(self, *_):
        pass


with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        for command in [["mojo", "run", "--Werror", "test.mojo"], ["./test-req"]]:
            subprocess.run(
                command,
                env={
                    **os.environ,
                    "REQ_PACKAGE_TEST_URL": f"http://127.0.0.1:{server.server_port}",
                },
                check=True,
                timeout=120,
            )
    finally:
        server.shutdown()
        worker.join()
