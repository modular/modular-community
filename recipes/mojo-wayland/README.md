# mojo-wayland

<!-- ![mojo_csv_logo](./mojo-wayland.jpeg) -->
<image src='./mojo-wayland.jpeg' width='900'/>

Mojo bindings for the Wayland client protocol, generated from the official
protocol XML.

## License

MIT — see [LICENSE](LICENSE).

The generated files under `wayland/gen/` are derived from the Wayland core
protocol and xdg-shell protocol XMLs, which carry their own permissive
copyright notices, reproduced in `wayland/c/generated/`. The generated
bindings inherit those notices.

## Versioning

This Project follows the compiler version its built against, not its own versioning. Any releases not tied to a compiler revision will be mark with v1, etc...

# AI Usage Notes

Ai tools are used to maintain documentation and notes, and will continue to be used for that purpose. However, EVERY single line of (non-script-generated) actual code is human reviewed and tested. That is the expectation for any contribution.

## Layout

| Path                       | Purpose                                                    |
| -------------------------- | ---------------------------------------------------------- |
| `scripts/wayland_bindgen.py` | Generator: protocol XML → Mojo modules (see below)         |
| `wayland/core.mojo`        | Hand-written runtime (external_call stubs, WLArgument, shim glue) — NOT regenerated |
| `wayland/gen/wayland.mojo` | Generated: requests, event opcodes, listen/next accessors  |
| `wayland/gen/xdg_shell.mojo` | Generated: xdg-shell (wm_base, surface, toplevel, ...)   |
| `wayland/c/shim.c`         | Tiny C shim: event capture dispatcher + interface table    |
| `wayland/c/generated/`     | wayland-scanner private-code + header for xdg-shell (xdg interfaces are NOT in libwayland-client) |
| `tests/test_pack.mojo`     | Headless packaged-library smoke test (no compositor)       |
| `tests/test_ffi_probe.mojo` | Executable FFI probe: zero-arg external_call status (live) |
| `tests/test_ffi_probe2.mojo` | Executable FFI probe: control + OwnedDLHandle (live)     |
| `tests/test_connect.mojo`  | Live test: connect → disconnect                            |
| `tests/test_globals.mojo`  | Live test: connect → registry → print all globals          |
| `tests/test_window.mojo`   | Live test: full xdg-shell window with a wl_shm gradient buffer |
| `tests/recipe.yaml`        | conda recipe for the modular-community channel             |
| `dist/`                    | Packed release artifacts (gitignored)                      |

## How it works

- **Requests** are lowered through `wl_proxy_marshal_array` /
  `wl_proxy_marshal_array_constructor_versioned` (libwayland exports no
  per-request symbols; `wl_surface_commit` & co. are header-inline only).
  Everything resolves via `external_call` against `libwayland-client`, linked
  at build time with `-lwayland-client`.
- **Interface records** (`wl_registry_interface` etc.) are data symbols — the
  C shim exposes a static name → `wl_interface*` table
  (`wayland_shim_interface("wl_registry")`).
- **Events** use one generic C dispatcher (`wl_proxy_add_dispatcher`) that
  captures `(opcode, args)` per proxy into a FIFO owned by the shim. Mojo
  polls with `{iface}_next_{event}(queue, out_args)` and frees copied string
  args with `_shim_string_free`.
- `WLArgument` mirrors `union wl_argument` exactly (**8 bytes on x86_64**).
- **Argument arrays are indexed by wire-signature position**: `new_id` slots
  stay zeroed (libwayland writes the new proxy id there); every other arg
  lands at its position in the XML arg list. Dense-packing non-`new_id` args
  causes the compositor to read garbage (e.g. `get_xdg_surface` signature
  `no` — the surface must be in slot 1, slot 0 is the new_id).
- **Child proxies inherit the parent's version**: constructors resolve the
  version via `wl_proxy_get_version(parent)` — never hardcode the XML version
  (binding `xdg_wm_base` at v3 and creating an `xdg_surface` at v7 is a
  protocol error).
- `wl_registry.bind` is special: the wire signature is `usun`, so the
  generated `wl_registry_bind` takes BOTH the interface pointer (for the
  constructed proxy) and the interface NAME string (marshalled as the `s`
  wire arg) plus name+version.
- **XDG interfaces** come from `wayland-scanner private-code` compiled into
  the shim DSO; they are not exported by libwayland-client.
- **Minimum libwayland version**: every C symbol the bindings call
  (`wl_proxy_marshal_array_constructor_versioned`, `wl_proxy_add_dispatcher`,
  `wl_proxy_get_version`, ...) has existed since wayland **1.10** (2016), so
  the practical floor is old. The conda recipe pins `wayland >=1.23` only
  because that is the oldest conda-forge version actually tested — see
  `[SYNC:minlib]` in `wayland/core.mojo` before adding a new stub.
- **Targets**: linux-64 and linux-aarch64 (both little-endian 64-bit;
  `sizeof(union wl_argument)` is 8 on each). Other Wayland platforms
  (big-endian or 32-bit) are unsupported by the `WLArgument` byte-cell
  emulation — see `[SYNC:abi]` in `wayland/core.mojo`.

## Using the library

Add the dependency to `pixi.toml` (after the package is published, or with a
path dep for local development):

```toml
[dependencies]
mojo-wayland = { path = "path/to/mojo-wayland" }   # or version once published
```

Because the bindings resolve shim symbols at load time, any binary using the
package must link the shim DSO alongside `libwayland-client` (path shown for
a conda/pixi env; use `-L/usr/lib` for the system libwayland):

```bash
mojo build app.mojo -I . \
    -Xlinker -L.pixi/envs/default/lib -Xlinker -lwayland-client \
    -Xlinker -L.pixi/lib -Xlinker -lwayland_shim
```

### Connect and enumerate globals

```mojo
from wayland.core import (
    WLPtr, WLArgument, MAX_EVENT_ARGS, _shim_string_free,
    wl_display_connect, wl_display_disconnect,
    wl_display_dispatch, wl_display_roundtrip, stack_allocation,
)
from wayland.gen.wayland import (
    wl_display_get_registry, wl_registry_listen, wl_registry_next_global,
)

def main() raises:
    var display = wl_display_connect(0)   # 0 = default socket ($WAYLAND_DISPLAY)
    if Int(display) == 0:
        raise Error("failed to connect to compositor")

    var registry = wl_display_get_registry(display)

    # install the capture dispatcher; the shim writes a queue handle to buf[0]
    var queue_buf = stack_allocation[1, WLPtr]()
    if wl_registry_listen(registry, queue_buf) != 0:
        raise Error("registry_listen failed")
    var queue = queue_buf[unsafe_offset=0]

    var args = stack_allocation[MAX_EVENT_ARGS, WLArgument]()  # MUST be 16 slots
    while wl_display_dispatch(display) > 0:
        while wl_registry_next_global(queue, args):
            # args[0]=name (u), args[1]=interface (s, malloc'd copy), args[2]=version (i)
            ...
```

### Bind a global and issue requests

Binding is generic (the wire signature is `usun`): pass the interface record
resolved via `shim_interface`, the interface name as a `WLString`, and the
global's name + negotiated version:

```mojo
from wayland.core import shim_interface
from wayland.gen.wayland import wl_registry_bind, wl_compositor_create_surface

var compositor = wl_registry_bind(
    registry, shim_interface("wl_compositor"),
    str_to_wlstring("wl_compositor"), name, version,
)
var surface = wl_compositor_create_surface(compositor)
wl_surface_commit(surface)
```

### Events: poll-based capture

Every interface with events gets `{iface}_listen(proxy, out_queue)` plus one
`{iface}_next_{event}(queue, out_args)` accessor per event. Poll after each
`wl_display_dispatch`; the accessors return `False` when no matching event is
pending. String (`s`) args are malloc'd copies — free them with
`_shim_string_free`:

```mojo
var evargs = stack_allocation[MAX_EVENT_ARGS, WLArgument]()
while xdg_toplevel_next_configure(top_queue, evargs):
    pass                          # reconfigure: keep current size
if xdg_toplevel_next_close(top_queue):
    running = False               # no args -> takes no out_args buffer
```

Decoding helpers for the raw `WLArgument` slots (byte-level, little-endian —
see `tests/test_window.mojo` for the full versions):

```mojo
def arg_as_uint(a: WLArgument) -> UInt32:
    var v = 0
    for i in range(4):
        v = v | (Int(a.raw[i]) << (8 * i))
    return UInt32(v)
```

### Full walkthrough

`tests/test_window.mojo` is a complete, working example (~340 lines) covering
the whole WSI path: registry → bind `wl_compositor`/`wl_shm`/`xdg_wm_base` →
surface chain → xdg configure/ack handshake → `memfd_create` + `mmap` +
`wl_shm` pool → ARGB8888 gradient → attach/damage/commit → event loop that
answers pings and exits on close. Read it top-to-bottom before writing your
own client; it is kept compilable and compositor-verified by CI.

### Memory and lifetime rules

- Opaque objects are raw `WLPtr` handles; there is **no** automatic lifetime
  management. Destructors are explicit generated functions
  (`wl_shm_pool_destroy(pool)` etc. — marshal + `wl_proxy_destroy`).
- Popped string args are owned by you: `_shim_string_free(ptr)` when done.
- Event arg buffers must be `MAX_EVENT_ARGS` (16) entries — the shim zeroes
  the whole array; a shorter stack buffer overflows.
- `memfd`/`mmap`-backed `wl_shm` buffers are plain libc calls via
  `external_call` (see the `_memfd_create`/`_mmap` bindings in
  `tests/test_window.mojo`); the library does not wrap them.

## Extending to other protocols

1. Pass the extension XML to the generator:
   `pixi run gen` (edit the task in `pixi.toml` to append more XMLs) — it
   emits `wayland/gen/{protocol}.mojo`.
2. Re-export the module in `wayland/__init__.mojo` (regenerated
   automatically by the generator's final write step).
3. Interfaces not exported by `libwayland-client` (like all of xdg-shell)
   need `wayland-scanner private-code` objects compiled into the shim — see
   the `scanner` and `shim` tasks — plus their `wl_*_interface` records
   added to `SHIM_IFACE_ENTRIES` in `wayland/c/shim.c`.
4. Check the generated opcodes against the XML: opcode = position of the
   request/event in the interface element (0-based).

## System dependencies

Dependencies are split into two pixi environments:

- **dev** (default) — `pixi run <task>` — day-to-day work: gen, scanner,
  shim, build, and the live tests. Provides Mojo + Python.
- **packaging** — `pixi run -e packaging <task>` — the conda-package
  workflow: `pack`, `test-pack`, `sync-recipe`, `test-build`. Adds
  `rattler-build` (conda-forge) on top of everything dev has. Install it
  with `pixi install -e packaging` (the dev env is the default).

The **host system** additionally needs (per task):

| Dependency | Needed by | Tasks | Notes |
| ---------- | --------- | ----- | ----- |
| C compiler (`gcc`) | dev + packaging | `shim`, `pack` | compiles `wayland/c/shim.c`; the conda recipe uses conda's own compiler instead |
| `wayland-scanner` | dev | `scanner` | ships with the `wayland` package |
| protocol XMLs (`wayland.xml`, `xdg-shell.xml`) | dev | `gen` | Arch: `wayland` + `wayland-protocols`; Debian: `libwayland-dev` + `wayland-protocols` |
| `libwayland-client` (+ headers) | dev | all test builds, `shim` | Debian: `libwayland-dev`; linked with `-lwayland-client` |
| Wayland compositor on `$WAYLAND_DISPLAY` | dev | live tests only | `test-connect`, `test-globals`, `test-window`, `test_ffi_probe*` |
| `sha256sum`, `awk` | packaging | `pack` | coreutils + gawk (present on any Linux) |
| `rattler-build` | packaging | `test-build` | conda-forge dep in this env, or system package |

Deployment (the published conda package) needs **none** of these on the
consumer host — `pixi install mojo-wayland` from a channel pulls
`libwayland-client` as a run-dependency automatically, and the recipe builds
in rattler-build's isolated env (conda gcc + conda wayland). The generated
outputs (`wayland/gen/`, `wayland/c/generated/`) are committed, so `gen` and
`scanner` are only required when protocols change.

Example install on Arch:

```bash
sudo pacman -S --needed wayland wayland-protocols
```

Debian/Ubuntu:

```bash
sudo apt install libwayland-dev wayland-protocols
```

## Tasks

```bash
pixi run gen          # regen Mojo bindings from protocol XML
pixi run shim         # build the C shim DSO into .pixi/lib/
pixi run build        # package the Mojo side → .pixi/envs/default/lib/wayland.mojoc
pixi run test-globals # build + run live registry test on $WAYLAND_DISPLAY
pixi run test-window  # build + run live xdg-shell window (400x300 gradient)
```

(Packaging tasks live in a separate `packaging` environment — see
[System dependencies](#system-dependencies) and
[Packaging](#packaging-modular-community).)

The registry test prints every global your compositor advertises and exits 0.
The window test opens a real 400x300 teal-purple gradient window (answers xdg
pings, exits on close), proving: connect, generic bind, xdg-shell configure
handshake, wl_shm via memfd, buffer attach/commit, and the event loop.

Note: on some pixi versions `pixi run` sandboxes the task environment in a way
that breaks `wl_display_connect` (connection refused at runtime). If the test
fails to connect under `pixi run` but builds fine, invoke the binary directly:

```bash
pixi run shim && pixi run build
pixi run mojo build tests/test_globals.mojo -I . -o .pixi/test_globals \
    -Xlinker -L.pixi/lib -Xlinker -L.pixi/envs/default/lib \
    -Xlinker -lwayland_shim -Xlinker -lwayland-client
LD_LIBRARY_PATH=.pixi/lib:.pixi/envs/default/lib .pixi/test_globals
```

## Packaging (modular-community)

```bash
pixi run -e packaging pack         # shim DSO + wayland.mojoc + sha256sum → dist/
pixi run -e packaging test-pack    # headless smoke test of dist/ artifacts (no compositor)
pixi run -e packaging sync-recipe  # copy recipe + test into ../modular-community/recipes/
pixi run -e packaging test-build   # full rattler-build build + conda test of tests/recipe.yaml
```

### Archival
The precompiled `.mojoc` we release against stable mojo-compiler updates for archival purposes.
This can also be used to pin against a specific version, just comment out the version check in pack.sh.

`pack` produces `dist/wayland.mojoc`, a precompiled artifact of the library.

- **The mojoc is compiler-version-locked, exactly.** The consuming compiler
  must be the *same build* it was compiled with. `pack` writes
  `dist/mojo.version` recording the producing compiler; check it before
  consuming.
- **The source tree is the universal fallback**: `-I <path-to-mojo-wayland>`
  works on any compiler that can parse it (the version error above
  disappears when importing from source) — that's how this repo's own
  tests consume the library.
- The shim DSO is a normal C shared object: no version lock, just
  `-Xlinker`/`-L`/`-lwayland_shim` at link time and `libwayland-client`
  at runtime (see [Using the library](#using-the-library)).

prefer the conda package (pixi/modular) when available


## Known limitations

- Events are poll-based (`dispatch` then `next_*` per event), not callback-
  based; a callback API would need C→Mojo trampolines.
- String args in popped events must be freed with `_shim_string_free`.
- The core protocol + xdg-shell are generated; further extensions need their
  XMLs passed to the generator (and any non-libwayland interfaces added to
  the shim's `SHIM_IFACE_ENTRIES` + scanner private-code build).
- Destructor requests marshal then `wl_proxy_destroy`; there is no
  automatic object-lifetime management.
