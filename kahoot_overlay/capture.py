import base64
import io
import logging
import threading
import time
from typing import Any, Dict, Tuple

import mss
from PIL import Image

# mss renamed its factory from mss.mss() to mss.MSS() in newer releases; support both.
_MSS_FACTORY = getattr(mss, "MSS", None) or getattr(mss, "mss")

from placement import nearest_monitor
from winutil import get_cursor_pos

logger = logging.getLogger("KahootOverlay.capture")


class ScreenCapturer:
    """
    Thread-safe screen capturer that reuses one thread-local mss instance
    per thread (satisfying mss Windows GDI thread-affinity rules).
    """

    def __init__(self):
        self._local = threading.local()

    def _get_sct(self) -> Any:
        sct = getattr(self._local, "sct", None)
        if sct is None:
            sct = _MSS_FACTORY()
            self._local.sct = sct
        return sct

    def get_monitor_at(self, x: int, y: int) -> Dict[str, int]:
        """Monitor containing the point (x, y); the nearest one if it is off-screen."""
        monitors = self._get_sct().monitors
        if len(monitors) <= 1:
            return dict(monitors[0])
        return dict(nearest_monitor(monitors[1:], x, y))

    def get_cursor_monitor(self) -> Dict[str, int]:
        cx, cy = get_cursor_pos()
        return self.get_monitor_at(cx, cy)

    def grab_monitor_raw(self) -> Tuple[Any, Dict[str, int], float]:
        """
        Grab the monitor containing the cursor.
        Returns (mss_screen_shot, monitor_rect_dict, capture_ms).
        """
        t0 = time.perf_counter()
        sct = self._get_sct()
        mon = self.get_cursor_monitor()
        raw_shot = sct.grab(mon)
        capture_ms = (time.perf_counter() - t0) * 1000.0
        return raw_shot, mon, capture_ms

    @staticmethod
    def encode_pil_image(
        img: Image.Image,
        max_edge: int = 1568,
        jpeg_quality: int = 80,
    ) -> Tuple[str, float, Tuple[int, int]]:
        """
        Resize PIL image so long edge <= max_edge and encode to base64 JPEG in memory.
        Returns (base64_str, encode_ms, (new_w, new_h)).
        """
        t0 = time.perf_counter()
        if img.mode != "RGB":
            img = img.convert("RGB")

        w, h = img.size
        long_edge = max(w, h)
        if long_edge > max_edge and max_edge > 0:
            scale = float(max_edge) / float(long_edge)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            img = img.resize((new_w, new_h), Image.LANCZOS)
        else:
            new_w, new_h = w, h

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_quality, optimize=False)
        b64_str = base64.b64encode(buf.getvalue()).decode("ascii")
        encode_ms = (time.perf_counter() - t0) * 1000.0
        return b64_str, encode_ms, (new_w, new_h)

    def encode_raw_shot(
        self,
        raw_shot: Any,
        max_edge: int = 1568,
        jpeg_quality: int = 80,
    ) -> Tuple[str, float, Tuple[int, int]]:
        t0 = time.perf_counter()
        img = Image.frombytes("RGB", raw_shot.size, raw_shot.bgra, "raw", "BGRX")
        b64_str, _, dims = self.encode_pil_image(img, max_edge=max_edge, jpeg_quality=jpeg_quality)
        encode_ms = (time.perf_counter() - t0) * 1000.0
        return b64_str, encode_ms, dims

    def load_file_and_encode(
        self,
        filepath: str,
        max_edge: int = 1568,
        jpeg_quality: int = 80,
    ) -> Tuple[str, float, float, Tuple[int, int]]:
        t0 = time.perf_counter()
        with Image.open(filepath) as im:
            img = im.convert("RGB")
        capture_ms = (time.perf_counter() - t0) * 1000.0
        b64_str, encode_ms, dims = self.encode_pil_image(img, max_edge=max_edge, jpeg_quality=jpeg_quality)
        return b64_str, capture_ms, encode_ms, dims

    def close(self) -> None:
        sct = getattr(self._local, "sct", None)
        if sct is not None:
            try:
                sct.close()
            except Exception:
                pass
            self._local.sct = None