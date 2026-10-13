"""Installed-package smoke test for the mcrypto conda package.

Runs against the precompiled ``mcrypto.mojoc`` installed under
``$PREFIX/lib/mojo`` rather than the source tree.
"""

from std.testing import (
    assert_equal,
    assert_raises,
    assert_true,
    assert_false,
    assert_not_equal,
)
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
from mcrypto.kdf.hkdf import extract, expand
from mcrypto.padding.pkcs7 import pad, unpad
from mcrypto.compression.transforms import zlib_compress, zlib_decompress
from mcrypto.kem import mlkem512
from mcrypto.key_exchange.x25519 import public_key as x25519_public_key
from mcrypto.signatures.xeddsa import sign_with_randomness, verify


def _hex(value: String) raises -> List[UInt8]:
    return transform(EncodingTransform.HEX_DECODE, value.as_bytes())


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
        len(
            authenticate(MacAlgorithm.CMAC, Span(mac_key), Span(plaintext), 16)
        ),
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

    # RFC 5869 test case 1, including its intermediate extraction result.
    var ikm = List[UInt8](length=22, fill=0x0B)
    var hkdf_salt = _hex("000102030405060708090a0b0c")
    var context = _hex("f0f1f2f3f4f5f6f7f8f9")
    var prk = extract(HashAlgorithm.SHA256, Span(hkdf_salt), Span(ikm))
    assert_equal(
        prk,
        _hex(
            "077709362c2e32df0ddc3f0dc47bba6390b6c73bb50f9c3122ec844ad7c2b3e5"
        ),
    )
    assert_equal(
        expand(HashAlgorithm.SHA256, Span(prk), Span(context), 42),
        _hex(
            "3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865"
        ),
    )

    # PKCS#7 bytes and rejection of inconsistent padding.
    var message = List("abc".as_bytes())
    var padded = pad(Span(message), 8)
    assert_equal(padded, _hex("6162630505050505"))
    assert_equal(unpad(Span(padded), 8), message)
    padded[3] ^= 1
    with assert_raises():
        _ = unpad(Span(padded), 8)

    # URL alphabet, omitted padding, and canonical trailing bits.
    var binary = _hex("fbff")
    assert_equal(
        transform(EncodingTransform.BASE64URL_NOPAD_ENCODE, Span(binary)),
        List("-_8".as_bytes()),
    )
    assert_equal(
        transform(EncodingTransform.BASE64URL_NOPAD_DECODE, "-_8".as_bytes()),
        binary,
    )
    with assert_raises():
        _ = transform(
            EncodingTransform.BASE64URL_NOPAD_DECODE, "-_9".as_bytes()
        )

    # Enforce the output budget in the installed decompressor.
    var compressed = zlib_compress(Span(message))
    assert_equal(zlib_decompress(Span(compressed), max_output_bytes=3), message)
    with assert_raises():
        _ = zlib_decompress(Span(compressed), max_output_bytes=2)

    # Exercise packaged ML-KEM-512 with fresh OS entropy and implicit rejection.
    var keys = mlkem512.keypair()
    var encapsulated = mlkem512.encapsulate(Span(keys[0]))
    assert_equal(len(keys[0]), 800)
    assert_equal(len(keys[1]), 1632)
    assert_equal(len(encapsulated[0]), 768)
    assert_equal(len(encapsulated[1]), 32)
    assert_equal(
        mlkem512.decapsulate(Span(encapsulated[0]), Span(keys[1])),
        encapsulated[1],
    )
    encapsulated[0][0] ^= 1
    var rejected = mlkem512.decapsulate(Span(encapsulated[0]), Span(keys[1]))
    assert_equal(len(rejected), 32)
    assert_not_equal(rejected, encapsulated[1])

    # Signal's published xeddsa_fast_test; fixed randomness is a test fixture.
    var signing_key = List[UInt8](length=32, fill=0)
    signing_key[8] = 189
    var signing_message = List[UInt8](length=200, fill=0)
    var signing_randomness = List[UInt8](length=64, fill=0)
    var public_key = x25519_public_key(Span(signing_key))
    assert_equal(
        public_key,
        _hex(
            "5995f464e9d34d5ca56b9905b9a3cc37c456b2d8d313edbcf584b705b5c04955"
        ),
    )
    var signature = sign_with_randomness(
        Span(signing_message), Span(signing_key), Span(signing_randomness)
    )
    assert_equal(
        signature,
        _hex(
            "11c7f3e6c4df9e8a5150e1db3b30f92de3a3b3aa438656545fa7390f4bcc7bb26c431d9e90643e4f0eaa0e9c557766fa69ada576d63dcaf2ac326c11d0b97702"
        ),
    )
    assert_true(
        verify(Span(signature), Span(signing_message), Span(public_key))
    )
    signing_message[0] ^= 1
    assert_false(
        verify(Span(signature), Span(signing_message), Span(public_key))
    )

    print("conda smoke tests: OK")
