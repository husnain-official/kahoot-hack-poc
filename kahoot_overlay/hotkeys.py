import ctypes
from ctypes import wintypes
import logging
import threading
from typing import Callable, Dict, Optional, Tuple

logger = logging.getLogger("KahootOverlay.hotkeys")

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

WM_HOTKEY = 0x0312
WM_QUIT = 0x0012

MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
MOD_NOREPEAT = 0x4000

HOTKEY_IDS = {
    "solve": 101,
    "toggle_overlay": 102,
    "quit": 103,
}

VK_MAP: Dict[str, int] = {
    "tab": 0x09,
    "enter": 0x0D,
    "return": 0x0D,
    "esc": 0x1B,
    "escape": 0x1B,
    "space": 0x20,
    "pageup": 0x21,
    "pagedown": 0x22,
    "end": 0x23,
    "home": 0x24,
    "left": 0x25,
    "up": 0x26,
    "right": 0x27,
    "down": 0x28,
    "insert": 0x2D,
    "delete": 0x2E,
    "del": 0x2E,
    "f1": 0x70,
    "f2": 0x71,
    "f3": 0x72,
    "f4": 0x73,
    "f5": 0x74,
    "f6": 0x75,
    "f7": 0x76,
    "f8": 0x77,
    "f9": 0x78,
    "f10": 0x79,
    "f11": 0x7A,
    "f12": 0x7B,
}


def parse_hotkey_string(hotkey_str: str) -> Optional[Tuple[int, int]]:
    """
    Parse a string like 'ctrl+alt+a' into (modifiers_with_MOD_NOREPEAT, virtual_key_code).
    Returns None if invalid.
    """
    if not hotkey_str or not isinstance(hotkey_str, str):
        return None
    parts = [p.strip().lower() for p in hotkey_str.split("+") if p.strip()]
    if not parts:
        return None

    mods = MOD_NOREPEAT
    vk: Optional[int] = None

    for part in parts:
        if part in ("ctrl", "control"):
            mods |= MOD_CONTROL
        elif part == "alt":
            mods |= MOD_ALT
        elif part == "shift":
            mods |= MOD_SHIFT
        elif part in ("win", "windows", "super"):
            mods |= MOD_WIN
        elif part in VK_MAP:
            vk = VK_MAP[part]
        elif len(part) == 1 and ("a" <= part <= "z" or "0" <= part <= "9"):
            vk = ord(part.upper())
        else:
            return None

    if vk is None:
        return None
    return mods, vk


class HotkeyListener:
    """
    Global hotkey manager using Win32 RegisterHotKey + GetMessageW loop on a dedicated thread.
    Requires no admin privileges and never steals window focus.
    """

    def __init__(
        self,
        hotkeys_cfg: Dict[str, str],
        on_action: Callable[[str], None],
        on_registration_error: Optional[Callable[[str, str], None]] = None,
    ):
        self.hotkeys_cfg = dict(hotkeys_cfg)
        self.on_action = on_action
        self.on_registration_error = on_registration_error
        self._thread: Optional[threading.Thread] = None
        self._thread_id: Optional[int] = None
        self._ready_event = threading.Event()
        self._registered_ids: list[int] = []

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._ready_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="Win32HotkeyThread", daemon=True)
        self._thread.start()
        self._ready_event.wait(timeout=2.0)

    def _run_loop(self) -> None:
        self._thread_id = kernel32.GetCurrentThreadId()
        id_to_action: Dict[int, str] = {}

        # Force message queue creation before signaling readiness
        msg = wintypes.MSG()
        user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 0)

        for action, hk_id in HOTKEY_IDS.items():
            hk_str = self.hotkeys_cfg.get(action, "")
            parsed = parse_hotkey_string(hk_str)
            if parsed is None:
                logger.warning("Invalid hotkey string '%s' for action '%s'.", hk_str, action)
                if self.on_registration_error:
                    self.on_registration_error(action, hk_str)
                continue

            mods, vk = parsed
            ok = user32.RegisterHotKey(None, hk_id, mods, vk)
            if ok:
                self._registered_ids.append(hk_id)
                id_to_action[hk_id] = action
                logger.info("Registered global hotkey '%s' for action '%s' (id=%d).", hk_str, action, hk_id)
            else:
                err = kernel32.GetLastError()
                logger.warning(
                    "RegisterHotKey failed for action '%s' ('%s'), win32 error=%d.",
                    action,
                    hk_str,
                    err,
                )
                if self.on_registration_error:
                    self.on_registration_error(action, hk_str)

        self._ready_event.set()

        try:
            while True:
                res = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
                if res <= 0:
                    break
                if msg.message == WM_HOTKEY:
                    hk_id = int(msg.wParam)
                    action = id_to_action.get(hk_id)
                    if action:
                        try:
                            self.on_action(action)
                        except Exception as exc:
                            logger.error("Error dispatching hotkey action %s: %s", action, exc)
                user32.TranslateMessage(ctypes.byref(msg))
                user32.DispatchMessageW(ctypes.byref(msg))
        finally:
            for hk_id in self._registered_ids:
                try:
                    user32.UnregisterHotKey(None, hk_id)
                except Exception:
                    pass
            self._registered_ids.clear()
            logger.info("Hotkey listener thread exited cleanly.")

    def stop(self) -> None:
        if self._thread_id is not None:
            try:
                user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
            except Exception as exc:
                logger.debug("PostThreadMessageW error: %s", exc)
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        self._thread = None
        self._thread_id = None