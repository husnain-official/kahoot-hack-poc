import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from placement import clamp_to_monitor, from_offset, nearest_monitor, to_offset

M1 = {"left": 0, "top": 0, "width": 1920, "height": 1080}
M2 = {"left": 1920, "top": 0, "width": 2560, "height": 1440}
LEFT = {"left": -1920, "top": 0, "width": 1920, "height": 1080}


class TestPlacement(unittest.TestCase):
    def test_nearest_monitor_inside(self):
        self.assertIs(nearest_monitor([M1, M2], 100, 100), M1)
        self.assertIs(nearest_monitor([M1, M2], 2000, 100), M2)

    def test_nearest_monitor_boundary_pixels(self):
        self.assertIs(nearest_monitor([M1, M2], 1919, 10), M1)
        self.assertIs(nearest_monitor([M1, M2], 1920, 10), M2)

    def test_nearest_monitor_offscreen_picks_closest(self):
        self.assertIs(nearest_monitor([M1, M2], 5000, 500), M2)
        self.assertIs(nearest_monitor([LEFT, M1], -3000, 500), LEFT)

    def test_nearest_monitor_empty_raises(self):
        with self.assertRaises(ValueError):
            nearest_monitor([], 0, 0)

    def test_clamp_keeps_window_fully_on_monitor(self):
        self.assertEqual(clamp_to_monitor(1800, 1000, 280, 124, M1), (1640, 956))
        self.assertEqual(clamp_to_monitor(-50, -20, 280, 124, M1), (0, 0))
        self.assertEqual(clamp_to_monitor(500, 400, 280, 124, M1), (500, 400))

    def test_clamp_on_secondary_and_negative_monitors(self):
        self.assertEqual(clamp_to_monitor(1800, 10, 280, 124, M2), (1920, 10))
        self.assertEqual(clamp_to_monitor(-2500, 10, 280, 124, LEFT), (-1920, 10))

    def test_clamp_window_larger_than_monitor_pins_top_left(self):
        tiny = {"left": 100, "top": 50, "width": 200, "height": 100}
        self.assertEqual(clamp_to_monitor(999, 999, 280, 124, tiny), (100, 50))

    def test_offset_round_trip(self):
        for mon in (M1, M2, LEFT):
            ox, oy = to_offset(mon["left"] + 33, mon["top"] + 44, mon)
            self.assertEqual((ox, oy), (33, 44))
            self.assertEqual(from_offset(ox, oy, mon), (mon["left"] + 33, mon["top"] + 44))


if __name__ == "__main__":
    unittest.main()
