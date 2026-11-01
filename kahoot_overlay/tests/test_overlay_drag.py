"""Drag / hide behaviour of OverlayWindow, using a stand-in for tkinter (no display needed)."""
import ctypes
import os
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if not hasattr(ctypes, "windll"):
    ctypes.windll = mock.MagicMock()
    ctypes.WinDLL = mock.MagicMock()
    ctypes.WINFUNCTYPE = mock.MagicMock()
if "tkinter" not in sys.modules:
    try:
        import tkinter  # noqa: F401
    except ImportError:
        sys.modules["tkinter"] = mock.MagicMock()

import overlay  # noqa: E402

M1 = {"left": 0, "top": 0, "width": 1920, "height": 1080}
M2 = {"left": 1920, "top": 0, "width": 2560, "height": 1440}


class FakeRoot:
    def __init__(self, x=400, y=50):
        self.x, self.y = x, y
        self.geometry_calls = []
        self.withdrawn = False

    def winfo_x(self):
        return self.x

    def winfo_y(self):
        return self.y

    def winfo_screenwidth(self):
        return 1920

    def winfo_screenheight(self):
        return 1080

    def geometry(self, spec):
        self.geometry_calls.append(spec)
        # Accept both "+x+y" and "WxH+x+y"; Tk writes negatives as "+-N".
        pos = spec[spec.index("+"):]
        x, y = pos[1:].split("+")
        self.x, self.y = int(x), int(y)

    def withdraw(self):
        self.withdrawn = True

    def update_idletasks(self):
        pass


def make_overlay(monitors=(M1, M2), root=None, **flags):
    ov = object.__new__(overlay.OverlayWindow)
    ov.root = root or FakeRoot()
    ov.width, ov.height = 280, 124
    ov.cfg = {"overlay": {"position": "top-right"}}
    ov.visible = True
    ov.draggable = flags.get("draggable", True)
    ov.click_through = flags.get("click_through", False)
    ov._auto_hide_after_id = None
    ov._last_monitor = None
    ov._drag_origin = None
    ov._drag_start_pointer = (0, 0)
    ov._drag_moved = False
    saved = []
    ov._on_position_changed = lambda ox, oy: saved.append((ox, oy))

    def monitor_at(px, py):
        from placement import nearest_monitor
        return nearest_monitor(list(monitors), px, py)

    ov._monitor_at = monitor_at
    ov._saved = saved
    return ov


def ev(x, y):
    return types.SimpleNamespace(x_root=x, y_root=y)


class TestDrag(unittest.TestCase):
    def test_drag_moves_window_and_saves_offset(self):
        ov = make_overlay()  # window at (400, 50)
        ov._on_drag_start(ev(500, 100))
        ov._on_drag_move(ev(700, 300))
        self.assertEqual((ov.root.x, ov.root.y), (600, 250))
        ov._on_drag_end(ev(700, 300))
        self.assertEqual(ov._saved, [(600, 250)])
        self.assertEqual(ov.cfg["overlay"]["position"], [600, 250])

    def test_plain_click_does_not_move_or_save(self):
        ov = make_overlay()
        ov._on_drag_start(ev(500, 100))
        ov._on_drag_move(ev(501, 101))  # below the 3px threshold
        ov._on_drag_end(ev(501, 101))
        self.assertEqual(ov._saved, [])
        self.assertEqual((ov.root.x, ov.root.y), (400, 50))

    def test_drop_past_right_edge_is_pulled_back_on_screen(self):
        ov = make_overlay(monitors=(M1,))
        ov._on_drag_start(ev(500, 100))
        ov._on_drag_move(ev(2000, 1200))
        ov._on_drag_end(ev(2000, 1200))
        self.assertEqual(ov._saved, [(1640, 956)])
        self.assertEqual((ov.root.x, ov.root.y), (1640, 956))

    def test_drop_on_second_monitor_saves_offset_from_that_monitor(self):
        ov = make_overlay()
        ov._on_drag_start(ev(500, 100))
        ov._on_drag_move(ev(2600, 400))  # window lands at x=2500, centre on monitor 2
        ov._on_drag_end(ev(2600, 400))
        self.assertEqual(ov._saved, [(580, 350)])  # 2500-1920, 400-50
        self.assertIs(ov._last_monitor, M2)

    def test_dragging_disabled_by_setting(self):
        ov = make_overlay(draggable=False)
        ov._on_drag_start(ev(500, 100))
        ov._on_drag_move(ev(900, 500))
        ov._on_drag_end(ev(900, 500))
        self.assertEqual((ov.root.x, ov.root.y), (400, 50))
        self.assertEqual(ov._saved, [])

    def test_dragging_disabled_when_click_through(self):
        ov = make_overlay(click_through=True)
        ov._on_drag_start(ev(500, 100))
        ov._on_drag_move(ev(900, 500))
        self.assertEqual((ov.root.x, ov.root.y), (400, 50))

    def test_save_failure_does_not_break_dragging(self):
        ov = make_overlay()
        ov._on_position_changed = mock.Mock(side_effect=OSError("disk full"))
        ov._on_drag_start(ev(500, 100))
        ov._on_drag_move(ev(700, 300))
        ov._on_drag_end(ev(700, 300))  # must not raise
        self.assertEqual((ov.root.x, ov.root.y), (600, 250))


class TestHideAndRestore(unittest.TestCase):
    def test_close_button_and_right_click_hide(self):
        for handler in ("_on_close_click", "_on_right_click"):
            ov = make_overlay()
            ov.hide = lambda ov=ov: setattr(ov, "visible", False)
            self.assertEqual(getattr(ov, handler)(ev(0, 0)), "break")
            self.assertFalse(ov.visible)

    def test_saved_position_is_relative_to_current_monitor(self):
        ov = make_overlay()
        ov.cfg = {"overlay": {"position": [100, 60]}}
        ov._position_on_monitor(M2)
        self.assertEqual((ov.root.x, ov.root.y), (2020, 60))

    def test_saved_position_off_screen_after_resolution_change_is_clamped(self):
        ov = make_overlay()
        ov.cfg = {"overlay": {"position": [5000, 4000]}}
        ov._position_on_monitor(M1)
        self.assertEqual((ov.root.x, ov.root.y), (1640, 956))

    def test_named_corner_still_works(self):
        ov = make_overlay()
        ov._position_on_monitor(M1)
        self.assertEqual((ov.root.x, ov.root.y), (1920 - 280 - 18, 18))


if __name__ == "__main__":
    unittest.main()
