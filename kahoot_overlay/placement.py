"""Pure geometry helpers for placing the overlay on a monitor (no tkinter / Win32)."""
from typing import Dict, Sequence, Tuple

Monitor = Dict[str, int]


def _rect(mon: Monitor) -> Tuple[int, int, int, int]:
    return int(mon["left"]), int(mon["top"]), int(mon["width"]), int(mon["height"])


def nearest_monitor(monitors: Sequence[Monitor], x: int, y: int) -> Monitor:
    """Monitor containing (x, y), or the closest one if the point is off every monitor."""
    best = None
    best_dist = 0
    for mon in monitors:
        left, top, width, height = _rect(mon)
        dx = max(left - x, 0, x - (left + width - 1))
        dy = max(top - y, 0, y - (top + height - 1))
        dist = dx * dx + dy * dy
        if best is None or dist < best_dist:
            best, best_dist = mon, dist
    if best is None:
        raise ValueError("no monitors available")
    return best


def clamp_to_monitor(x: int, y: int, width: int, height: int, mon: Monitor, margin: int = 0) -> Tuple[int, int]:
    """Move a width x height window so it lies fully inside the monitor.

    If the window is larger than the monitor it is pinned to the top-left corner.
    """
    left, top, mon_w, mon_h = _rect(mon)
    min_x, min_y = left + margin, top + margin
    max_x = max(left + mon_w - width - margin, min_x)
    max_y = max(top + mon_h - height - margin, min_y)
    return min(max(x, min_x), max_x), min(max(y, min_y), max_y)


def to_offset(x: int, y: int, mon: Monitor) -> Tuple[int, int]:
    """Absolute screen position -> offset from the monitor's top-left corner."""
    left, top, _, _ = _rect(mon)
    return x - left, y - top


def from_offset(ox: int, oy: int, mon: Monitor) -> Tuple[int, int]:
    """Offset from the monitor's top-left corner -> absolute screen position."""
    left, top, _, _ = _rect(mon)
    return left + ox, top + oy
