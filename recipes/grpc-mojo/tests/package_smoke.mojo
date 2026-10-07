from std.testing import assert_equal, assert_true

from grpc import (
    GrpcChannel,
    Health,
    Metadata,
    PollingServer,
    PollingServerConfig,
    Server,
    ServerContext,
    Status,
    StatusCode,
    decode_timeout,
    encode_timeout,
    frame_message,
    status_code_name,
)
from net import resolve
from proto import ProtoMessage, WireReader, WireWriter, decode, encode
from tls import PeerCertificate, SubjectAlternativeNames, TLSContext


struct PackageMessage(Copyable, Defaultable, Movable, ProtoMessage):
    var value: String
    var _unknown: List[Byte]

    def __init__(out self):
        self.value = String()
        self._unknown = List[Byte]()

    def __init__(out self, var value: String):
        self.value = value^
        self._unknown = List[Byte]()

    def encode_to(self, mut writer: WireWriter):
        if self.value.byte_length() != 0:
            writer.string_field(1, self.value)
        writer.buf.extend(Span(self._unknown))

    def merge_from(mut self, mut reader: WireReader) raises:
        while not reader.done():
            var tag = reader.read_tag()
            if tag[0] == 1:
                self.value = reader.string_value()
            else:
                reader.capture_field(tag[0], tag[1], self._unknown)


def _check_client_identity_api() raises:
    """Type-checks the installed channel API without opening a connection."""
    _ = GrpcChannel.connect_tls(
        "localhost",
        443,
        cert_chain_pem="client-chain.pem",
        key_pem="client.key",
    )


def _check_server_identity_api() raises:
    """Type-checks required client authentication without binding."""
    _ = Server.tls(
        "127.0.0.1",
        443,
        "server-chain.pem",
        "server.key",
        client_ca_file="client-ca.pem",
        require_client_cert=True,
    )


def _check_polling_server_identity_api() raises:
    """Type-checks polling server client authentication without binding."""
    _ = PollingServer.tls(
        "127.0.0.1",
        443,
        "server-chain.pem",
        "server.key",
        PollingServerConfig(),
        client_ca_file="client-ca.pem",
        require_client_cert=True,
    )


def _check_peer_certificate_api() raises:
    """Type-checks the owned client identity exposed to handlers."""
    var unauthenticated = ServerContext()
    assert_true(not unauthenticated.peer_certificate)

    var der: List[Byte] = [0x30, 0x00]
    var certificate = PeerCertificate(
        leaf_der=der^,
        verified=True,
        matched_name=String(),
        subject_alt_names=SubjectAlternativeNames(
            dns_names=["client.example.test"],
            uri_names=["spiffe://example.test/client"],
            email_addresses=["client@example.test"],
            ip_addresses=["192.0.2.44", "2001:db8::44"],
        ),
    )
    var peer: Optional[PeerCertificate] = certificate^
    var authenticated = ServerContext(peer)
    assert_true(authenticated.peer_certificate)
    assert_true(authenticated.peer_certificate.value().verified)
    assert_equal(len(authenticated.peer_certificate.value().leaf_der), 2)
    assert_equal(
        authenticated.peer_certificate.value().subject_alt_names.dns_names[0],
        "client.example.test",
    )
    assert_equal(
        authenticated.peer_certificate.value().subject_alt_names.uri_names[0],
        "spiffe://example.test/client",
    )
    assert_equal(
        authenticated.peer_certificate.value().subject_alt_names.email_addresses[
            0
        ],
        "client@example.test",
    )
    assert_equal(
        authenticated.peer_certificate.value().subject_alt_names.ip_addresses[
            1
        ],
        "2001:db8::44",
    )


def main() raises:
    assert_true(Status.ok().is_ok())
    assert_equal(
        String(status_code_name(StatusCode.UNAVAILABLE)), "UNAVAILABLE"
    )
    assert_equal(encode_timeout(1_500), "1500n")
    assert_equal(decode_timeout("2m"), 2_000_000)

    var encoded = encode(PackageMessage(value="installed"))
    var decoded = decode[PackageMessage](Span(encoded))
    assert_equal(decoded.value, "installed")

    var framed = frame_message(Span(encoded))
    assert_equal(len(framed), len(encoded) + 5)
    assert_equal(framed[0], 0)

    var metadata = Metadata()
    metadata.add("x-package", "installed")
    assert_equal(metadata.get("x-package").value(), "installed")
    assert_equal(len(metadata.entries), 1)

    var addresses = resolve("127.0.0.1", 443)
    assert_true(len(addresses) >= 1)
    assert_equal(addresses[0].port, 443)

    _ = TLSContext.client(verify=False, alpn=["h2"])
    _check_peer_certificate_api()
    var health = Health()
    health.set_status("echo.Echo", 1)
    assert_true(health.status("echo.Echo"))
    print("grpc-mojo package smoke test passed")
