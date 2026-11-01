import logging
from typing import Callable, Optional

from PIL import Image, ImageDraw
import pystray
from pystray import Menu, MenuItem

from config import ConfigManager

logger = logging.getLogger("KahootOverlay.tray")

TRIES_CHOICES = (1, 2, 3, 5, 7, 9)


def create_tray_icon_image(size: int = 64) -> Image.Image:
    """Generate the 4-color Kahoot tile icon in memory using Pillow."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    half = size // 2
    pad = 2

    # Top-left: Red
    draw.rectangle([pad, pad, half - pad, half - pad], fill=(226, 27, 60, 255))
    # Top-right: Blue
    draw.rectangle([half + pad, pad, size - pad, half - pad], fill=(19, 104, 206, 255))
    # Bottom-left: Yellow
    draw.rectangle([pad, half + pad, half - pad, size - pad], fill=(216, 158, 0, 255))
    # Bottom-right: Green
    draw.rectangle([half + pad, half + pad, size - pad, size - pad], fill=(38, 137, 12, 255))

    return img


class TrayManager:
    """System tray icon and menu using pystray.run_detached()."""

    def __init__(
        self,
        config_mgr: ConfigManager,
        on_solve: Callable[[], None],
        on_toggle_overlay: Callable[[], None],
        on_open_config: Callable[[], None],
        on_open_log: Callable[[], None],
        on_quit: Callable[[], None],
        on_reset_overlay_position: Optional[Callable[[], None]] = None,
        on_overlay_settings_changed: Optional[Callable[[], None]] = None,
    ):
        self.config_mgr = config_mgr
        self.on_solve = on_solve
        self.on_toggle_overlay = on_toggle_overlay
        self.on_open_config = on_open_config
        self.on_open_log = on_open_log
        self.on_quit = on_quit
        self.on_reset_overlay_position = on_reset_overlay_position
        self.on_overlay_settings_changed = on_overlay_settings_changed
        self.icon: Optional[pystray.Icon] = None

    def _make_tries_item(self, n: int) -> MenuItem:
        def on_select(icon, item):
            self.config_mgr.update(lambda c: c.update({"tries": n}))
            logger.info("Tray updated tries -> %d", n)

        def is_checked(item) -> bool:
            return int(self.config_mgr.get().get("tries", 3)) == n

        return MenuItem(str(n), on_select, checked=is_checked, radio=True)

    def _make_mode_item(self, label: str, mode_val: str) -> MenuItem:
        def on_select(icon, item):
            self.config_mgr.update(lambda c: c.update({"mode": mode_val}))
            logger.info("Tray updated mode -> %s", mode_val)

        def is_checked(item) -> bool:
            return self.config_mgr.get().get("mode", "balanced") == mode_val

        return MenuItem(label, on_select, checked=is_checked, radio=True)

    def _make_pipeline_item(self, label: str, pipe_val: str) -> MenuItem:
        def on_select(icon, item):
            self.config_mgr.update(lambda c: c.update({"pipeline": pipe_val}))
            logger.info("Tray updated pipeline -> %s", pipe_val)

        def is_checked(item) -> bool:
            return self.config_mgr.get().get("pipeline", "single") == pipe_val

        return MenuItem(label, on_select, checked=is_checked, radio=True)

    def _is_draggable(self, item=None) -> bool:
        return bool(self.config_mgr.get().get("overlay", {}).get("draggable", True))

    def _toggle_draggable(self, icon, item) -> None:
        new_val = not self._is_draggable()
        self.config_mgr.update(lambda c: c.setdefault("overlay", {}).update({"draggable": new_val}))
        logger.info("Tray updated overlay.draggable -> %s", new_val)
        if self.on_overlay_settings_changed:
            self.on_overlay_settings_changed()

    def _build_menu(self) -> Menu:
        tries_submenu = Menu(*(self._make_tries_item(n) for n in TRIES_CHOICES))
        mode_submenu = Menu(
            self._make_mode_item("Fast", "fast"),
            self._make_mode_item("Balanced", "balanced"),
        )
        pipeline_submenu = Menu(
            self._make_pipeline_item("Single-stage", "single"),
            self._make_pipeline_item("Two-stage", "two_stage"),
        )

        overlay_submenu = Menu(
            MenuItem("Allow dragging", self._toggle_draggable, checked=self._is_draggable),
            MenuItem(
                "Reset position",
                lambda icon, item: self.on_reset_overlay_position() if self.on_reset_overlay_position else None,
            ),
        )

        return Menu(
            MenuItem("Solve now", lambda icon, item: self.on_solve(), default=True),
            MenuItem("Show/Hide overlay", lambda icon, item: self.on_toggle_overlay()),
            Menu.SEPARATOR,
            MenuItem("Tries", tries_submenu),
            MenuItem("Mode", mode_submenu),
            MenuItem("Pipeline", pipeline_submenu),
            MenuItem("Overlay", overlay_submenu),
            Menu.SEPARATOR,
            MenuItem("Open config file", lambda icon, item: self.on_open_config()),
            MenuItem("Open log file", lambda icon, item: self.on_open_log()),
            Menu.SEPARATOR,
            MenuItem("Quit", lambda icon, item: self.on_quit()),
        )

    def start(self) -> None:
        image = create_tray_icon_image()
        self.icon = pystray.Icon(
            "KahootOverlay",
            image,
            "Kahoot Answer Overlay",
            menu=self._build_menu(),
        )
        self.icon.run_detached()
        logger.info("System tray icon started via run_detached().")

    def notify(self, title: str, message: str) -> None:
        if self.icon is None:
            return
        try:
            self.icon.notify(message[:240], title[:60])
        except Exception as exc:
            logger.debug("Tray notification failed: %s", exc)

    def stop(self) -> None:
        if self.icon is not None:
            try:
                self.icon.stop()
            except Exception as exc:
                logger.debug("Error stopping tray icon: %s", exc)
            self.icon = None