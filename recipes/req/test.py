"""Exercise the installed package using the source release smoke test."""

import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path("tests").resolve()))
from conftest import Fixtures  # noqa: E402


with Fixtures() as fixtures:
    env = dict(os.environ, **fixtures.environment)
    for name in ("REQ_NATIVE_LIB", "REQ_EXPECT_LOAD_ERROR", "REQ_EXPECT_HTTP2_UNAVAILABLE"):
        env.pop(name, None)
    for command in [
        ["mojo", "run", "--Werror", "tests/package.mojo"],
        ["./test-req"],
    ]:
        subprocess.run(command, env=env, check=True, timeout=120)
