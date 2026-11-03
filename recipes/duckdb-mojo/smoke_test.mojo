from duckdb import *
from std.testing import assert_equal


def main() raises:
    # Offline smoke test for the packaged bindings: open an in-memory database,
    # run a trivial query, and verify the result. Exercises the dlopen of
    # libduckdb (the runtime dependency) without touching the network.
    var con = DuckDB.connect(":memory:")
    var result = con.execute("SELECT 40 + 2 AS answer")
    assert_equal(result.fetch_chunk().get[Int32](col=0, row=0), 42)
