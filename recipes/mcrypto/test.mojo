"""Installed-package smoke test for the mcrypto conda package.

Runs against the precompiled ``mcrypto.mojoc`` installed under
``$PREFIX/lib/mojo`` rather than the source tree.
"""

from std.testing import assert_equal, assert_raises
from mcrypto.aead.algorithm import AeadAlgorithm
from mcrypto.aead.combined import decrypt, encrypt
from mcrypto.encoding.algorithm import EncodingTransform
from mcrypto.encoding.transforms import transform
from mcrypto.hashes import HashAlgorithm, hash
from mcrypto.hashes.xof import xof
from mcrypto.hashes.xof_algorithm import XofAlgorithm
from mcrypto.kdf.algorithm import KdfAlgorithm
from mcrypto.kdf.dispatch import derive
from mcrypto.macs.algorithm import MacAlgorithm
from mcrypto.macs.dispatch import authenticate


def main() raises:
    # Fixed hash against a known vector.
    assert_equal(
        hash(HashAlgorithm.SHA256, "abc".as_bytes()),
        transform(
            EncodingTransform.HEX_DECODE,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
            .as_bytes(),
        ),
    )

    # Extendable-output function against a known vector.
    assert_equal(
        xof(XofAlgorithm.SHAKE128, "abc".as_bytes(), 32),
        transform(
            EncodingTransform.HEX_DECODE,
            "5881092dd818bf5cf8a3ddb793fbcba74097d5c526a6d35f97b83351940f2cc8"
            .as_bytes(),
        ),
    )

    # AEAD round trip and tamper detection.
    var key = List[UInt8](length=32, fill=0x35)
    var nonce = List[UInt8](length=24, fill=0x46)
    var aad = List("conda smoke".as_bytes())
    var plaintext = List("authenticated payload".as_bytes())
    var sealed = encrypt(
        AeadAlgorithm.XCHACHA20_POLY1305_IETF,
        Span(key),
        Span(nonce),
        Span(aad),
        Span(plaintext),
    )
    assert_equal(
        decrypt(
            AeadAlgorithm.XCHACHA20_POLY1305_IETF,
            Span(key),
            Span(nonce),
            Span(aad),
            Span(sealed),
        ),
        plaintext,
    )
    sealed[len(sealed) - 1] ^= 1
    with assert_raises():
        _ = decrypt(
            AeadAlgorithm.XCHACHA20_POLY1305_IETF,
            Span(key),
            Span(nonce),
            Span(aad),
            Span(sealed),
        )

    # Keyed authentication.
    var mac_key = List[UInt8](length=16, fill=0x71)
    assert_equal(
        len(authenticate(MacAlgorithm.CMAC, Span(mac_key), Span(plaintext), 16)),
        16,
    )

    # Deterministic key derivation is reproducible and context-separated.
    var secret = List("input key material".as_bytes())
    var salt = List("conda salt".as_bytes())
    var info = List("mcrypto conda smoke".as_bytes())
    assert_equal(
        derive(
            KdfAlgorithm.HKDF_SHA256, Span(secret), Span(salt), Span(info), 32
        ),
        derive(
            KdfAlgorithm.HKDF_SHA256, Span(secret), Span(salt), Span(info), 32
        ),
    )

    print("conda smoke tests: OK")
