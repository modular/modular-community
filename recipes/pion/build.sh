#!/usr/bin/env bash
# Builds pion-server as Pion's release workflow does at this tag (the `build`
# task in pixi.toml on macOS and linux-aarch64, `build-portable` on linux-64),
# with the target CPU pinned so the binary runs on any machine of its platform.
set -euo pipefail

case "${target_platform}" in
  osx-arm64)     vec=macos-arm64;   cpu=apple-m1 ;;
  linux-64)      vec=linux-x86_64;  cpu=x86-64-v2 ;;
  linux-aarch64) vec=linux-aarch64; cpu=neoverse-n1 ;;
  *) echo "unsupported platform: ${target_platform}" >&2; exit 1 ;;
esac

c_shims=(crash_wrap fcntl_wrap geo_math uring_wrap xdp_wrap ssm_state_wrap
         moe_warm_pool build_pool_wrap worker_spawn_wrap)
link=()
for f in crash_wrap build_pool_wrap worker_spawn_wrap fcntl_wrap geo_math \
         uring_wrap xdp_wrap; do
  link+=(-Xlinker "src/ffi/$f.o")
done

if [[ "${target_platform}" == osx-* ]]; then
  c_shims+=(dev_macos_stubs)
  link+=(-Xlinker src/ffi/dev_macos_stubs.o)
fi
for f in ssm_state_wrap moe_warm_pool; do
  link+=(-Xlinker "src/ffi/$f.o")
done
for f in "${c_shims[@]}"; do
  gcc -O3 -c "src/ffi/$f.c" -o "src/ffi/$f.o"
done

if [[ "${target_platform}" == osx-* ]]; then
  for f in metal_wrap nle_wrap; do
    clang -O3 -c -fobjc-arc "src/ffi/$f.m" -o "src/ffi/$f.o"
    link+=(-Xlinker "src/ffi/$f.o")
  done
fi

bash scripts/build_lua.sh
link+=(-Xlinker src/ffi/lua_wrap.o -Xlinker src/ffi/lua/liblua.a)

if [[ "${target_platform}" == osx-* ]]; then
  link+=(-Xlinker -export_dynamic
         -Xlinker -u -Xlinker _pion_gh199_build_lane
         -Xlinker -u -Xlinker _pion_worker_entry
         -Xlinker -u -Xlinker _pion_script_dispatch
         -Xlinker -framework -Xlinker Metal
         -Xlinker -framework -Xlinker Foundation
         -Xlinker -framework -Xlinker NaturalLanguage
         -Xlinker "vendor/pion-vector/${vec}/libpion_vector.a")
else
  link+=(-Xlinker --export-dynamic
         -Xlinker -u -Xlinker pion_gh199_build_lane
         -Xlinker -u -Xlinker pion_worker_entry
         -Xlinker -u -Xlinker pion_script_dispatch
         -Xlinker "vendor/pion-vector/${vec}/libpion_vector.a"
         -Xlinker -lm -Xlinker -lpthread)
fi

# Stamp src/common/version.mojo from VERSION, as `pixi run build` does. The
# committed copy can lag VERSION (at v0.9.7 it still said 0.9.6), and a tag
# archive has no .git, so the stamp reads the commit from .export_sha. GIT_DIR
# keeps git from finding a repository the build directory happens to sit in.
echo "${PION_COMMIT}" > .export_sha
GIT_DIR=/nonexistent bash scripts/stamp_version.sh

mkdir -p "${PREFIX}/bin"
mojo build -I . -D PION_HELD_VECTOR -O3 --target-cpu "${cpu}" src/main.mojo \
  "${link[@]}" -o "${PREFIX}/bin/pion-server"

# The --metal-attention shader. pion-server looks for it next to its own
# executable, including <exe>/../share/pion. Compiling it needs full Xcode;
# without it the flag reports NOT ACTIVE and nothing else changes.
if [[ "${target_platform}" == osx-* ]]; then
  if xcrun -sdk macosx metal -c src/ffi/metal_compute.metal -o metal_compute.air &&
     xcrun -sdk macosx metallib metal_compute.air -o metal_compute.metallib; then
    mkdir -p "${PREFIX}/share/pion"
    cp metal_compute.metallib "${PREFIX}/share/pion/"
  else
    echo "[Metal] no shader compiler (needs full Xcode): --metal-attention will report NOT ACTIVE" >&2
  fi
fi

# install_name_tool tries and fails to rewrite paths inside these.
find "${PREFIX}" -type d -name .mojo_cache -prune -exec rm -rf {} +
# The compiler writes these into the prefix it runs from; they are not Pion's.
rm -rf "${PREFIX}/share/max/crashdb" "${PREFIX}/share/max/firstActivation"
