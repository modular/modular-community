"""Smoke tests for the mojo-window package. Runs headless via SDL dummy driver."""

from std.os import setenv
from std.testing import TestSuite, assert_true, assert_false, assert_equal, assert_raises
from window import (
    Window,
    GLWindow,
    Event,
    Quit,
    Resized,
    KeyDown,
    KeyUp,
    MouseMoved,
    MouseButtonDown,
    MouseButtonUp,
    MouseWheel,
)


def test_window_open_and_close() raises -> None:
    var w = Window("t", 100, 100)
    assert_true(w.is_open())
    w.close()
    assert_false(w.is_open())


def test_window_poll_returns_no_events() raises -> None:
    var w = Window("t", 64, 64)
    var events = w.events()
    assert_true(len(events) == 0)


def test_gl_window_fails_cleanly_under_dummy_driver() raises -> None:
    with assert_raises(contains="SDL_CreateWindow failed"):
        var w = GLWindow("t", 64, 64)


def test_event_variants() raises -> None:
    assert_true((Event(Quit())).isa[Quit]())
    var r: Event = Resized(800, 600)
    assert_equal(r[Resized].width, 800)
    assert_equal(r[Resized].height, 600)
    var kd: Event = KeyDown(42)
    assert_equal(kd[KeyDown].keycode, 42)
    var ku: Event = KeyUp(7)
    assert_equal(ku[KeyUp].keycode, 7)
    var mm: Event = MouseMoved(12, 34)
    assert_equal(mm[MouseMoved].x, 12)
    var mbd: Event = MouseButtonDown(1, 5, 6)
    assert_equal(mbd[MouseButtonDown].button, 1)
    var mbu: Event = MouseButtonUp(2, 9, 10)
    assert_equal(mbu[MouseButtonUp].button, 2)
    var mw: Event = MouseWheel(0, -1)
    assert_equal(mw[MouseWheel].y, -1)


def main() raises:
    _ = setenv("SDL_VIDEODRIVER", "dummy", True)
    TestSuite.discover_tests[__functions_in_module()]().run()
