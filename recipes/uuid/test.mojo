from uuid import Generator, UUID, Version, Variant


def main() raises:
    var generator = Generator()

    var uuid_v4 = generator.v4()
    var uuid_v4_str = uuid_v4.to_string()
    UUID.validate(uuid_v4_str)

    if uuid_v4.version() != Version(Version.v4):
        raise Error("Expected v4 UUID version bits")

    if uuid_v4.variant() != Variant(Variant.RFC9562):
        raise Error("Expected RFC9562 variant for v4 UUID")

    var uuid_v7 = generator.v7()
    var uuid_v7_str = uuid_v7.to_string()
    UUID.validate(uuid_v7_str)

    if uuid_v7.version() != Version(Version.v7):
        raise Error("Expected v7 UUID version bits")

    if uuid_v7.variant() != Variant(Variant.RFC9562):
        raise Error("Expected RFC9562 variant for v7 UUID")

    var parsed = UUID.from_string(uuid_v7_str)
    if parsed.to_string() != uuid_v7_str:
        raise Error("Round-trip parse/format failed")

    print("Conda package smoke test passed")
