"""Smoke test for the installed safetensors-mojo Conda package."""

from std.testing import assert_equal, assert_true

from safetensors import SafeDType, decode_header_length, parse_metadata


def main() raises:
    var header = '{"tensor":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}'
    var header_length = UInt64(header.byte_length())
    var contents = List[UInt8]()

    for index in range(8):
        contents.append(
            UInt8((header_length >> UInt64(index * 8)) & UInt64(0xFF))
        )
    for byte in header.as_bytes():
        contents.append(byte)
    contents.append(17)
    contents.append(29)

    assert_equal(decode_header_length(contents), header_length)

    var metadata = parse_metadata(contents)
    assert_equal(len(metadata), 1)
    assert_true(metadata.contains("tensor"))
    assert_equal(metadata.data_length(), UInt64(2))

    var tensor = metadata.info("tensor")
    assert_equal(tensor.dtype, SafeDType.U8)
    assert_equal(len(tensor.shape), 1)
    assert_equal(tensor.shape[0], UInt64(2))
    assert_equal(tensor.begin, UInt64(0))
    assert_equal(tensor.end, UInt64(2))
    assert_equal(tensor.element_count, UInt64(2))
    assert_equal(tensor.bit_length, UInt64(16))
    assert_equal(tensor.byte_length, UInt64(2))

    print("safetensors-mojo installed-package smoke test passed")
