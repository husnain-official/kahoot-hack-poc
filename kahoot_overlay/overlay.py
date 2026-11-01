import ctypes
import logging
import tkinter as tk
from typing import Any, Callable, Dict, List, Optional, Tuple

from placement import clamp_to_monitor, to_offset
from winutil import apply_overlay_exstyles

logger = logging.getLogger("KahootOverlay.overlay")

user32 = ctypes.windll.user32

TILE_COLORS = {
    "red": "#E21B3C",
    "blue": "#1368CE",
    "yellow": "#D89E00",
    "green": "#26890C",
}

TILE_GLYPHS = {
    "red": "▲",
    "blue": "◆",
    "yellow": "●",
    "green": "■",
}

BG_DARK = "#18191C"
BG_THINKING = "#252830"
BG_ERROR = "#5A1A22"
FG_WHITE = "#FFFFFF"
FG_DIM = "#B8BCC4"

# Minimum size so long messages (errors) always fit; smaller config values are raised to this.
MIN_WIDTH = 280
MIN_HEIGHT = 124


class OverlayWindow:
    """
    Borderless, always-on-top, non-activating (WS_EX_NOACTIVATE), toolwindow (WS_EX_TOOLWINDOW)
    tkinter overlay. No animations or fades; updates happen immediately on the main thread.
    """

    def __init__(
        self,
        root: tk.Tk,
        cfg: Dict[str, Any],
        on_position_changed: Optional[Callable[[int, int], None]] = None,
        monitor_at: Optional[Callable[[int, int], Dict[str, int]]] = None,
    ):
        self.root = root
        self.cfg = cfg
        self._on_position_changed = on_position_changed
        self._monitor_at = monitor_at
        self._drag_origin: Optional[Tuple[int, int]] = None
        self._drag_start_pointer: Tuple[int, int] = (0, 0)
        self._drag_moved: bool = False
        self.visible: bool = True
        self._auto_hide_after_id: Optional[str] = None
        self._last_monitor: Optional[Dict[str, int]] = None

        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)

        ov_cfg = cfg.get("overlay", {})
        self.width = max(int(ov_cfg.get("width", 280)), MIN_WIDTH)
        self.height = max(int(ov_cfg.get("height", 124)), MIN_HEIGHT)
        self.opacity = float(ov_cfg.get("opacity", 0.92))
        self.click_through = bool(ov_cfg.get("click_through", False))
        self.draggable = bool(ov_cfg.get("draggable", True))

        self.root.attributes("-alpha", self.opacity)
        self.root.configure(bg=BG_DARK, highlightthickness=1, highlightbackground="#3A3D45")

        # Line 1: Answer banner (color + shape glyph)
        self.banner_frame = tk.Frame(self.root, bg=BG_THINKING, height=38)
        self.banner_frame.pack(fill=tk.X, side=tk.TOP)
        self.banner_frame.pack_propagate(False)

        self.lbl_line1 = tk.Label(
            self.banner_frame,
            text="KAHOOT OVERLAY READY",
            font=("Segoe UI", 12, "bold"),
            fg=FG_WHITE,
            bg=BG_THINKING,
            anchor="center",
        )
        self.lbl_line1.pack(fill=tk.BOTH, expand=True, padx=6, pady=2)

        # Small close button (top-right). Right-clicking anywhere on the overlay also hides it.
        self.lbl_close = tk.Label(
            self.banner_frame,
            text="\u2715",
            font=("Segoe UI", 10, "bold"),
            fg="#D0D3DA",
            bg=BG_THINKING,
            cursor="hand2",
        )
        self.lbl_close.place(relx=1.0, x=-6, y=3, anchor="ne")
        self.lbl_close.bind("<ButtonPress-1>", self._on_close_click)

        # Body container for Lines 2, 3, 4
        self.body_frame = tk.Frame(self.root, bg=BG_DARK)
        self.body_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=3)

        self.lbl_line2 = tk.Label(
            self.body_frame,
            text="Press Ctrl+Alt+A to solve",
            font=("Segoe UI", 9, "bold"),
            fg=FG_WHITE,
            bg=BG_DARK,
            anchor="w",
        )
        self.lbl_line2.pack(fill=tk.X)

        self.lbl_line3 = tk.Label(
            self.body_frame,
            text="Confidence: --",
            font=("Segoe UI", 9),
            fg=FG_WHITE,
            bg=BG_DARK,
            anchor="w",
        )
        self.lbl_line3.pack(fill=tk.X)

        self.lbl_line4 = tk.Label(
            self.body_frame,
            text="Idle",
            font=("Segoe UI", 8),
            fg=FG_DIM,
            bg=BG_DARK,
            anchor="w",
        )
        self.lbl_line4.pack(fill=tk.X)

        self._bind_drag_and_hide()
        self._apply_wrap()
        self._position_on_monitor(None)
        self.root.update_idletasks()
        self.hwnd = self._get_hwnd()
        apply_overlay_exstyles(self.hwnd, click_through=self.click_through)

    # ------------------------------------------------------------------ drag / hide
    def _bind_drag_and_hide(self) -> None:
        widgets = (
            self.root, self.banner_frame, self.lbl_line1,
            self.body_frame, self.lbl_line2, self.lbl_line3, self.lbl_line4,
        )
        for w in widgets:
            w.bind("<ButtonPress-1>", self._on_drag_start, add="+")
            w.bind("<B1-Motion>", self._on_drag_move, add="+")
            w.bind("<ButtonRelease-1>", self._on_drag_end, add="+")
            w.bind("<ButtonPress-3>", self._on_right_click, add="+")
            try:
                w.configure(cursor="fleur")
            except tk.TclError:
                pass

    def _on_close_click(self, _event: Any) -> str:
        self.hide()
        return "break"

    def _on_right_click(self, _event: Any) -> str:
        self.hide()
        return "break"

    def _on_drag_start(self, event: Any) -> None:
        if not self.draggable or self.click_through:
            self._drag_origin = None
            return
        self._drag_origin = (event.x_root - self.root.winfo_x(), event.y_root - self.root.winfo_y())
        self._drag_start_pointer = (event.x_root, event.y_root)
        self._drag_moved = False

    def _on_drag_move(self, event: Any) -> None:
        if self._drag_origin is None:
            return
        if not self._drag_moved:
            dx = abs(event.x_root - self._drag_start_pointer[0])
            dy = abs(event.y_root - self._drag_start_pointer[1])
            if dx < 3 and dy < 3:
                return  # a plain click, not a drag
            self._drag_moved = True
        x = event.x_root - self._drag_origin[0]
        y = event.y_root - self._drag_origin[1]
        self.root.geometry(f"+{x}+{y}")

    def _on_drag_end(self, _event: Any) -> None:
        was_dragging = self._drag_origin is not None and self._drag_moved
        self._drag_origin = None
        self._drag_moved = False
        if was_dragging:
            self._commit_position()

    def _commit_position(self) -> None:
        """Snap fully onto the monitor it was dropped on and remember the spot."""
        x, y = self.root.winfo_x(), self.root.winfo_y()
        mon: Optional[Dict[str, int]] = None
        if self._monitor_at is not None:
            try:
                mon = self._monitor_at(x + self.width // 2, y + self.height // 2)
            except Exception as exc:
                logger.debug("Monitor lookup failed after drag: %s", exc)
        if mon is None:
            mon = {
                "left": 0,
                "top": 0,
                "width": self.root.winfo_screenwidth(),
                "height": self.root.winfo_screenheight(),
            }
        x, y = clamp_to_monitor(x, y, self.width, self.height, mon)
        self.root.geometry(f"+{x}+{y}")
        ox, oy = to_offset(x, y, mon)
        self._last_monitor = mon
        self.cfg.setdefault("overlay", {})["position"] = [ox, oy]
        if self._on_position_changed is not None:
            try:
                self._on_position_changed(ox, oy)
            except Exception as exc:
                logger.warning("Could not save overlay position: %s", exc)

    def _apply_wrap(self) -> None:
        """Wrap body lines to the window width instead of clipping them."""
        wrap = max(120, self.width - 24)
        for lbl in (self.lbl_line2, self.lbl_line3, self.lbl_line4):
            lbl.configure(wraplength=wrap, justify="left")

    def _set_banner(self, text: str, bg: str) -> None:
        """Set the big top line, shrinking the font for longer text so it never clips."""
        n = len(text)
        size = 12 if n <= 20 else (10 if n <= 28 else 9)
        self.lbl_line1.configure(text=text, bg=bg, fg=FG_WHITE, font=("Segoe UI", size, "bold"))
        self.lbl_close.configure(bg=bg)

    def _get_hwnd(self) -> int:
        try:
            frame_id = self.root.wm_frame()
            if frame_id and frame_id != "0x0":
                return int(frame_id, 16)
        except Exception:
            pass
        return int(self.root.winfo_id())

    def apply_config(self, cfg: Dict[str, Any], monitor: Optional[Dict[str, int]] = None) -> None:
        self.cfg = cfg
        ov_cfg = cfg.get("overlay", {})
        self.width = max(int(ov_cfg.get("width", 280)), MIN_WIDTH)
        self.height = max(int(ov_cfg.get("height", 124)), MIN_HEIGHT)
        self.opacity = float(ov_cfg.get("opacity", 0.92))
        self.draggable = bool(ov_cfg.get("draggable", True))
        self._apply_wrap()
        new_ct = bool(ov_cfg.get("click_through", False))

        if self.visible:
            self.root.attributes("-alpha", self.opacity)
        if new_ct != self.click_through:
            self.click_through = new_ct
            self.hwnd = self._get_hwnd()
            apply_overlay_exstyles(self.hwnd, click_through=self.click_through)

        self._position_on_monitor(monitor or self._last_monitor)

    def _position_on_monitor(self, monitor: Optional[Dict[str, int]]) -> None:
        if monitor is not None:
            self._last_monitor = monitor
            m_left = int(monitor.get("left", 0))
            m_top = int(monitor.get("top", 0))
            m_width = int(monitor.get("width", self.root.winfo_screenwidth()))
            m_height = int(monitor.get("height", self.root.winfo_screenheight()))
        else:
            m_left = 0
            m_top = 0
            m_width = self.root.winfo_screenwidth()
            m_height = self.root.winfo_screenheight()

        ov_cfg = self.cfg.get("overlay", {})
        pos = ov_cfg.get("position", "top-right")
        margin = 18
        explicit = False

        if isinstance(pos, list) and len(pos) == 2:
            explicit = True
            x = m_left + int(pos[0])
            y = m_top + int(pos[1])
        else:
            pos_str = str(pos).lower().strip()
            if pos_str == "top-left":
                x = m_left + margin
                y = m_top + margin
            elif pos_str == "bottom-left":
                x = m_left + margin
                y = m_top + m_height - self.height - margin - 40
            elif pos_str == "bottom-right":
                x = m_left + m_width - self.width - margin
                y = m_top + m_height - self.height - margin - 40
            elif "," in pos_str:
                explicit = True
                try:
                    parts = [int(p.strip()) for p in pos_str.split(",", 1)]
                    x = m_left + parts[0]
                    y = m_top + parts[1]
                except Exception:
                    x = m_left + m_width - self.width - margin
                    y = m_top + margin
            else:
                # Default top-right
                x = m_left + m_width - self.width - margin
                y = m_top + margin

        if explicit:
            x, y = clamp_to_monitor(
                x, y, self.width, self.height,
                {"left": m_left, "top": m_top, "width": m_width, "height": m_height},
            )
        self.root.geometry(f"{self.width}x{self.height}+{x}+{y}")

    def hide_for_capture_sync(self) -> bool:
        """
        Immediately make the overlay invisible before screen capture so it never
        contaminates the screenshot. Costs < 2 ms via layered window alpha=0 + DwmFlush.
        Returns True if the window was visible prior to hiding.
        """
        was_visible = self.visible
        if not was_visible:
            return False
        self.root.attributes("-alpha", 0.0)
        self.root.update_idletasks()
        try:
            ctypes.windll.dwmapi.DwmFlush()
        except Exception:
            pass
        return True

    def restore_after_capture_thinking(
        self,
        was_visible: bool,
        show_on_solve: bool,
        monitor: Optional[Dict[str, int]],
    ) -> None:
        """Immediately display 'Thinking…' right after the screen grab completes."""
        self._cancel_auto_hide()
        self._position_on_monitor(monitor)
        self.set_thinking_state()

        if was_visible or show_on_solve:
            if not self.visible:
                self.root.deiconify()
                self.hwnd = self._get_hwnd()
                apply_overlay_exstyles(self.hwnd, click_through=self.click_through)
                self.visible = True
            self.root.attributes("-alpha", self.opacity)
            self.root.update_idletasks()

    def set_thinking_state(self) -> None:
        self._cancel_auto_hide()
        self.banner_frame.configure(bg=BG_THINKING)
        self._set_banner("Thinking…", BG_THINKING)
        self.lbl_line2.configure(text="Analyzing screen…")
        self.lbl_line3.configure(text="Confidence: …")
        self.lbl_line4.configure(text="Working…")

    def show_result(self, result: Dict[str, Any], total_ms: float, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self._cancel_auto_hide()

        ov_cfg = cfg.get("overlay", {})
        if not self.visible and bool(ov_cfg.get("show_on_solve", True)):
            self.show()

        status = result.get("status", "error")
        ms_int = int(round(total_ms))

        if status == "ok" and result.get("found"):
            answers: List[str] = result.get("answer", ["red"])
            positions: List[str] = result.get("pos", ["top-left"])
            conf: int = int(result.get("confidence", 50))
            agree_text: str = result.get("agree_text", "1 of 1 tries agree")

            primary_color = answers[0].lower() if answers else "red"
            banner_bg = TILE_COLORS.get(primary_color, BG_THINKING)

            parts = []
            for c in answers:
                c_low = c.lower()
                glyph = TILE_GLYPHS.get(c_low, "")
                parts.append(f"{c_low.upper()} {glyph}".strip())
            line1_text = " + ".join(parts)

            pos_text = " + ".join(positions) if positions else "center"

            self.banner_frame.configure(bg=banner_bg)
            self._set_banner(line1_text, banner_bg)
            self.lbl_line2.configure(text=pos_text)
            self.lbl_line3.configure(text=f"Confidence: {conf}%")
            self.lbl_line4.configure(text=f"{agree_text} · {ms_int} ms")

        elif status == "no_question":
            self.banner_frame.configure(bg=BG_THINKING)
            self._set_banner("No Kahoot question found", BG_THINKING)
            self.lbl_line2.configure(text="Position: --")
            self.lbl_line3.configure(text="Confidence: --")
            self.lbl_line4.configure(text=f"Checked in {ms_int} ms")

        else:
            err_reason = str(result.get("error", "Unknown error"))
            self.show_error(err_reason, total_ms=total_ms)
            return

        self._schedule_auto_hide_if_configured()

    def show_error(self, short_reason: str, total_ms: Optional[float] = None) -> None:
        self._cancel_auto_hide()
        self.banner_frame.configure(bg=BG_ERROR)
        self._set_banner("ERROR", BG_ERROR)
        self.lbl_line2.configure(text=short_reason.strip()[:90])
        self.lbl_line3.configure(text="Details: tray menu > Open log file")
        if total_ms is not None:
            self.lbl_line4.configure(text=f"Failed · {int(round(total_ms))} ms")
        else:
            self.lbl_line4.configure(text="Ready for next hotkey")
        self._schedule_auto_hide_if_configured()

    def show(self) -> None:
        """Show overlay with last result preserved in < 10 ms using deiconify()."""
        self._cancel_auto_hide()
        self.root.deiconify()
        self.root.attributes("-alpha", self.opacity)
        self.root.attributes("-topmost", True)
        self.hwnd = self._get_hwnd()
        apply_overlay_exstyles(self.hwnd, click_through=self.click_through)
        self.visible = True
        self.root.update_idletasks()

    def hide(self) -> None:
        """Hide overlay immediately using withdraw()."""
        self._cancel_auto_hide()
        self.root.withdraw()
        self.visible = False
        self.root.update_idletasks()

    def toggle(self) -> bool:
        if self.visible:
            self.hide()
        else:
            self.show()
        return self.visible

    def _cancel_auto_hide(self) -> None:
        if self._auto_hide_after_id is not None:
            try:
                self.root.after_cancel(self._auto_hide_after_id)
            except Exception:
                pass
            self._auto_hide_after_id = None

    def _schedule_auto_hide_if_configured(self) -> None:
        ov_cfg = self.cfg.get("overlay", {})
        auto_sec = float(ov_cfg.get("auto_hide_seconds", 0.0))
        if auto_sec > 0.0:
            ms = int(round(auto_sec * 1000.0))
            self._auto_hide_after_id = self.root.after(ms, self.hide)