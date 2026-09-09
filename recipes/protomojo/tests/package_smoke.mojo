from std.testing import assert_equal

from any_pb import Any
from package_smoke_pb import PackagePayload, PackageSmoke, SmokeState
from package_smoke_pb_json_resolver import json_type_resolver
from proto import (
    JsonParseOptions,
    JsonPrintOptions,
    decode,
    decode_json,
    encode,
    encode_json,
)


def main() raises:
    var sent = PackageSmoke()
    sent.request_id = 42
    sent.payload = "installed package"
    sent.state = SmokeState(value=SmokeState.SMOKE_STATE_READY)
    var payload = PackagePayload()
    payload.note = "resolved from installed codegen"
    var metadata = Any()
    metadata.type_url = "type.googleapis.com/PackagePayload"
    metadata.value = encode(payload)
    sent.metadata = metadata^

    var received = decode[PackageSmoke](Span(encode(sent)))
    assert_equal(received.request_id, 42)
    assert_equal(received.payload, "installed package")
    assert_equal(received.state.value, SmokeState.SMOKE_STATE_READY)
    assert_equal(
        decode[PackagePayload](Span(received.metadata.value().value)).note,
        "resolved from installed codegen",
    )

    var print_options = JsonPrintOptions(
        type_resolver=json_type_resolver()
    )
    var parse_options = JsonParseOptions(
        type_resolver=json_type_resolver()
    )
    var text = encode_json(sent, options=print_options)
    var json_received = decode_json[PackageSmoke](
        text, options=parse_options
    )
    assert_equal(json_received.request_id, 42)
    assert_equal(json_received.payload, "installed package")
    assert_equal(json_received.state.value, SmokeState.SMOKE_STATE_READY)
    assert_equal(
        decode[PackagePayload](
            Span(json_received.metadata.value().value)
        ).note,
        "resolved from installed codegen",
    )
    print("protomojo package smoke test passed")
