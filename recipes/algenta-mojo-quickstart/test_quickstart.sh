#!/usr/bin/env bash
# Test for the algenta-mojo-quickstart conda package (modular-community).
#
# Two layers, both must pass:
#   1. package contents — the quickstart sources are installed where promised
#   2. end-to-end demo — copy the packaged sources into a scratch directory
#      and run `pixi run --locked demo`: resolves the Mojo toolchain and the
#      signed algenta-runtime-native wheel, compiles main.mojo, and executes
#      the full runtime round-trip. This mirrors exactly what a user does.
set -euo pipefail

QUICKSTART="$PREFIX/share/algenta-mojo-quickstart"

echo "== layer 1: package contents =="
for f in main.mojo pixi.toml pixi.lock README.md; do
    test -f "$QUICKSTART/$f"
    echo "  found: $QUICKSTART/$f"
done

echo "== layer 2: end-to-end demo from the packaged sources =="
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cp -r "$QUICKSTART" "$WORK/algenta-mojo-quickstart"
cd "$WORK/algenta-mojo-quickstart"

# Compile with the toolchain from the test environment first, so a broken
# source fails fast with a compiler error rather than a resolver error.
mojo build -o "$WORK/compile-check" main.mojo
test -x "$WORK/compile-check"
echo "  main.mojo compiles cleanly with the max toolchain from the test env"

# Full user path: pinned pixi workspace → signed runtime wheel → demo.
pixi run --locked demo

echo "algenta-mojo-quickstart package test: OK"
