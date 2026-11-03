# Package test for the wgpu-mojo conda package.
#
# rattler-build runs this in a fresh environment containing only the package
# and its declared runtime dependencies -- no repo checkout, no -I flag, no
# LD_LIBRARY_PATH from a dev tree. So it fails if any of these is untrue:
#
#   1. lib/mojo/wgpu.mojoc is where the Mojo compiler looks for it
#      (otherwise `from wgpu import ...` does not resolve)
#   2. libwgpu_native and libwgpu_mojo_cb are both installed and loadable
#      (check_symbols() constructs a WGPULib, which dlopens both)
#   3. the installed wgpu-native exports the ABI this binding targets
#      (check_symbols() probes the drift-prone entry points by name)
#
# Deliberately no adapter request: this must pass on a build machine with no
# GPU and no display server.
from wgpu import Instance
from wgpu.diagnostics import check_symbols


def main() raises:
    var missing = check_symbols()
    if len(missing) != 0:
        var msg = String(
            "installed wgpu-native is missing "
        ) + String(len(missing)) + " critical symbols:"
        for name in missing:
            msg += "\n  " + name
        raise Error(msg)
    print("wgpu-mojo package test OK — package resolves, native libs load, ABI matches")
