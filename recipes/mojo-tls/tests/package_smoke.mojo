from std.ffi import c_int, external_call
from std.sys import argv
from std.testing import assert_equal, assert_true

from net import TCPListener, TCPStream
from tls import PeerCertificate, SubjectAlternativeNames, TLSContext


def run_server(
    cert: String, key: String, client_ca: String
) raises -> Tuple[UInt16, c_int]:
    var listener = TCPListener("127.0.0.1", 0)
    var port = listener.local_port
    var pid = external_call["fork", c_int]()
    if pid == 0:
        try:
            var context = TLSContext.server(
                cert,
                key,
                client_ca_file=client_ca,
                require_client_cert=True,
                alpn=["h2"],
            )
            var tcp = listener.accept()
            var stream = context.accept(tcp^)
            var peer = stream.peer_certificate()
            if not peer:
                raise Error("installed server did not receive client identity")
            var identity: PeerCertificate = peer.value().copy()
            if not identity.verified or len(identity.leaf_der) == 0:
                raise Error("installed server did not verify client identity")
            if identity.matched_name != "":
                raise Error("server client identity had a matched hostname")
            var client_names: SubjectAlternativeNames = (
                identity.subject_alt_names.copy()
            )
            if (
                len(client_names.dns_names) != 1
                or client_names.dns_names[0] != "client.example.test"
                or client_names.uri_names[0]
                != "spiffe://example.test/package-client"
                or client_names.email_addresses[0] != "client@example.test"
                or client_names.ip_addresses[0] != "192.0.2.45"
            ):
                raise Error("installed server copied wrong client names")
            var request = stream.read_exact(4)
            stream.write_all(Span(request))
            stream.close()
        except:
            external_call["_exit", NoneType](c_int(1))
        external_call["_exit", NoneType](c_int(0))
    listener.close()
    return (port, pid)


def main() raises:
    var args = argv()
    assert_equal(
        len(args),
        6,
        "expected CA, server identity, and client identity paths",
    )

    var compatibility_der = List[Byte](length=1, fill=0x30)
    var compatibility = PeerCertificate(
        compatibility_der^, False, String("legacy.example")
    )
    assert_equal(compatibility.matched_name, "legacy.example")
    assert_equal(len(compatibility.subject_alt_names.dns_names), 0)

    var server = run_server(
        String(args[2]), String(args[3]), String(args[1])
    )
    var context = TLSContext.client(
        ca_file=String(args[1]),
        cert_chain_pem=String(args[4]),
        key_pem=String(args[5]),
        alpn=["h2"],
    )
    var tcp = TCPStream.connect("127.0.0.1", server[0])
    var stream = context.connect(tcp^, "localhost")
    var peer = stream.peer_certificate()
    if not peer:
        raise Error("installed client did not receive server identity")
    var identity: PeerCertificate = peer.value().copy()
    assert_true(identity.verified)
    assert_true(len(identity.leaf_der) > 0)
    assert_equal(identity.matched_name, "localhost")
    assert_equal(identity.subject_alt_names.dns_names[0], "localhost")
    assert_equal(
        identity.subject_alt_names.uri_names[0],
        "spiffe://example.test/package-server",
    )
    assert_equal(
        identity.subject_alt_names.email_addresses[0],
        "server@example.test",
    )
    assert_equal(identity.subject_alt_names.ip_addresses[0], "127.0.0.1")

    assert_true(stream.version().startswith("TLSv1."), stream.version())
    assert_equal(stream.negotiated_alpn(), "h2")
    stream.write_all("ping".as_bytes())
    assert_equal(String(from_utf8=stream.read_exact(4)), "ping")
    stream.close()
    assert_true(len(identity.leaf_der) > 0)
    assert_equal(
        identity.subject_alt_names.uri_names[0],
        "spiffe://example.test/package-server",
    )

    var status = c_int(0)
    var waited = external_call["waitpid", c_int](
        server[1], Pointer(to=status), c_int(0)
    )
    assert_equal(waited, server[1], "waitpid failed")
    assert_equal(status, 0, "TLS server failed")
    print("installed mojo-tls package handshake passed")
