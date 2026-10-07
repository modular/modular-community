# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Josh S Wilkinson
# Headless smoke test for the packaged `wayland` library (no compositor needed).
#
# Verifies the C shim resolves protocol interface records by name, exercising
# both libwayland_shim.so and the scanner-generated xdg-shell private code
# compiled into it. Run with the shim preloaded:
#
#   LD_PRELOAD=$PREFIX/lib/libwayland_shim.so \
#     mojo run -I $PREFIX/lib/mojo/wayland.mojopkg test_pack.mojo
from wayland.core import shim_interface


def main() raises:
    var iface = shim_interface("wl_registry")
    if Int(iface) == 0:
        raise Error("shim_interface(wl_registry) returned NULL")

    # xdg-shell interfaces are NOT exported by libwayland-client; they come
    # from wayland-scanner private code compiled into the shim DSO itself.
    var xdg = shim_interface("xdg_wm_base")
    if Int(xdg) == 0:
        raise Error("shim_interface(xdg_wm_base) returned NULL")

    print("✅ wayland package smoke test PASSED")
