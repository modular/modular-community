# mojo-websockets

`mojo-websockets` is a lightweight library for handling WebSocket connections in Mojo. 

It aims to provide an similar interface as the [python-websockets](https://github.com/python-websockets/websockets) package for creating WebSocket servers and clients, with additional features for enhanced usability.

## Disclaimer ⚠️

This software is in a early stage of development, using the Mojo nightly version. Please DO NOT use yet for production ready services.

## Features

- **WebSocket Server and Client**: Supports creating both WebSocket servers and clients.
- **Compatibility**: API designed to be intuitive for developers familiar with the Python websockets library.
- **Sans/IO Layer**: Implements a WebSocket Sans/IO layer pure Mojo and performs no I/O of its own.

For a complete listing, see the [features](https://github.com/msaelices/mojo-websockets/docs/features.md) document.

## Installation

1. **Install [magic](https://docs.modular.com/magic#install-magic)**

2. **Add the WebSockets Package** (at the top level of your project):

    ```bash
    magic add websockets
    ```
## Example of usage

### Server side

```mojo
from websockets.aliases import Bytes
from websockets.sync.server import serve, WSConnection
from websockets.utils.bytes import bytes_to_str


fn on_message(conn: WSConnection, data: Bytes) raises -> None:
    str_received = bytes_to_str(data)
    print("<<< ", str_received)
    conn.send_text(str_received)
    print(">>> ", str_received)


fn main() raises:
    with serve(on_message, "127.0.0.1", 8000) as server:
        server.serve_forever()
```

### Client side

```mojo
from websockets.sync.client import connect
from websockets.utils.bytes import bytes_to_str


fn send_and_receive(msg: String) raises:
    with connect("ws://127.0.0.1:8000") as client:
        client.send_text(msg)
        print(">>> ", msg)
        response = client.recv()
        print("<<< ", bytes_to_str(response))

fn main() raises:
    send_and_receive("Hello world!")

```

## TODO

- [ ] Asynchronous non-blocking communication (waiting for the Mojo async/await support)
- [ ] Implement automatic reconnection for clients
- [ ] Get rid of Python dependencies and logic (e.g. no more `from python import ...`)
- [ ] Make sure it passes all the tests in [Autobahn|Testsuite](https://github.com/crossbario/autobahn-testsuite/)
- [ ] Implement subprotocols and extensions
- [ ] Optimize performance for high-concurrency scenarios
- [ ] TLS support

See all the remaining features in the [features](https://github.com/msaelices/mojo-websockets/docs/features.md) document.

## Contributing

Contributions are welcome! If you'd like to contribute, please follow the contribution guidelines in the [CONTRIBUTING.md](https://github.com/msaelices/mojo-websockets/CONTRIBUTING.md) file in the repository.

## Acknowledgments

We have taken a lot of code from the amazing [lightbug_http](https://github.com/saviorand/lightbug_http) project.

Also, we took inspiration and some code from the [python-websockets](https://github.com/websockets) project, specially for implementing the [WebSocket Sans/IO layer](https://websockets.readthedocs.io/en/stable/howto/sansio.html) and their tests.

## License

mojo-websockets is licensed under the [MIT license](https://github.com/msaelices/mojo-websockets/LICENSE).
