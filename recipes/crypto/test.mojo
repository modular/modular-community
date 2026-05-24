from crypto.hashes import md5, sha1


def main() raises:
    var md5_digest = md5("abc".as_bytes())
    assert md5_digest.to_hex() == "900150983cd24fb0d6963f7d28e17f72", "MD5 mismatch"

    var sha1_digest = sha1("abc".as_bytes())
    assert sha1_digest.to_hex() == "a9993e364706816aba3e25717850c26c9cd0d89d", "SHA-1 mismatch"

    print("conda smoke tests: OK")
