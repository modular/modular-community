from difpatch import apply_patch, unified_diff


def main() raises:
    var old = "a\nb\nc\n"
    var new = "a\nB\nc\n"
    if apply_patch(old, unified_diff(old, new)) != new:
        raise Error("round-trip failed")
