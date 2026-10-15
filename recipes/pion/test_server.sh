#!/usr/bin/env bash
# Starts the packaged server and runs the two standard-library checks Pion's
# release workflow runs before packaging: the wire basics (test_raw.py) and a
# 1536-dim recall check that only a correct vector build passes.
set -euo pipefail

expected_version="$1"

pion-server --version | tee version.txt
grep -q "^pion-server ${expected_version}+" version.txt
# The tuned kernels must be linked, not the open reference.
grep -q "libpion_vector .*(closed" version.txt

port=$((20000 + RANDOM % 20000))
pion-server -p "${port}" -w 1 --no-auto-detect --no-auto-embed > server.log 2>&1 &
pid=$!
trap 'kill "${pid}" 2>/dev/null || true; wait "${pid}" 2>/dev/null || true' EXIT

for _ in $(seq 1 60); do
  if python -c "import socket, sys; socket.create_connection(('127.0.0.1', int(sys.argv[1])), timeout=1)" "${port}" 2>/dev/null; then
    break
  fi
  if ! kill -0 "${pid}" 2>/dev/null; then
    cat server.log
    exit 1
  fi
  sleep 1
done

python tests/test_raw.py --port "${port}" || { cat server.log; exit 1; }
python tests/test_vector_recall_smoke.py --port "${port}" || { cat server.log; exit 1; }
