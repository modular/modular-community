from net import (
    IPv4Address,
    Poller,
    TCPListener,
    TCPStream,
    Wakeup,
    is_would_block,
)


def main() raises:
    var address = IPv4Address("127.0.0.1", 443)
    if address.port != 443:
        raise Error("installed mojo-net package returned the wrong port")

    var listener = TCPListener("127.0.0.1", 0)
    listener.set_nonblocking(True)
    var poller = Poller()
    poller.register(listener.descriptor(), readable=True, writable=False)
    var client = TCPStream.connect("127.0.0.1", listener.local_port)
    if len(poller.wait(5000)) == 0:
        raise Error("installed listener did not become readable")
    var accepted = listener.accept()
    if not accepted.nonblocking:
        raise Error("accepted stream did not inherit non-blocking mode")
    var blocked = False
    try:
        _ = accepted.read_exact(1)
    except e:
        if not is_would_block(e):
            raise e
        blocked = True
    if not blocked:
        raise Error("empty accepted stream did not report would-block")
    client.close()
    accepted.close()
    poller.close()
    listener.close()

    var wakeup = Wakeup()
    var wake_poller = Poller()
    wake_poller.register(wakeup.descriptor(), readable=True, writable=False)
    wakeup.notify()
    if len(wake_poller.wait(5000)) == 0:
        raise Error("installed Wakeup did not wake Poller.wait")
    wakeup.drain()
    wakeup.close()
    wake_poller.close()
    print("mojo-net package smoke test passed")
