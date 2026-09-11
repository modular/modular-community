from std.testing import assert_equal

from h2 import H2_ALPN, FrameHeader
from hpack import Decoder, HeaderField


def main() raises:
    var header = FrameHeader(
        length=1024, frame_type=0x1, flags=0x5, stream_id=7
    )
    var wire = List[Byte]()
    header.serialize(wire)
    var parsed = FrameHeader.parse(Span(wire))
    assert_equal(parsed.length, 1024)
    assert_equal(parsed.frame_type, 0x1)
    assert_equal(parsed.flags, 0x5)
    assert_equal(parsed.stream_id, 7)

    var block = List[Byte]()
    block.append(0x82)
    block.append(0x86)
    block.append(0x84)
    var decoder = Decoder()
    var fields = decoder.decode(Span(block))
    assert_equal(len(fields), 3)
    assert_equal(fields[0], HeaderField(name=":method", value="GET"))
    assert_equal(fields[1], HeaderField(name=":scheme", value="http"))
    assert_equal(fields[2], HeaderField(name=":path", value="/"))
    assert_equal(String(H2_ALPN), "h2")
    print("installed mojo-http2 package smoke test passed")
