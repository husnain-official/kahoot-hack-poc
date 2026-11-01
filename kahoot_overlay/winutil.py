import ctypes
from ctypes import wintypes
import sys
import os
import logging

logger = logging.getLogger("KahootOverlay.winutil")

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

GWL_EXSTYLE = -20
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_LAYERED = 0x00080000
WS_EX_TOPMOST = 0x00000008

HWND_TOPMOST = -1
SWP_NOMOVE = 0x0002
SWP_NOSIZE = 0x0001
SWP_NOACTIVATE = 0x0010
SWP_SHOWWINDOW = 0x0040
SWP_HIDEWINDOW = 0x0080
SWP_FRAMECHANGED = 0x0020

ERROR_ALREADY_EXISTS = 183


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


def set_dpi_awareness() -> None:
    """Set Per-Monitor V2 DPI awareness before any window creation."""
    try:
        # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4
        res = user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        if res:
            return
    except Exception as exc:
        logger.debug("SetProcessDpiAwarenessContext failed: %s", exc)

    try:
        shcore = ctypes.windll.shcore
        # PROCESS_PER_MONITOR_DPI_AWARE = 2
        res = shcore.SetProcessDpiAwareness(2)
        if res == 0:
            return
    except Exception as exc:
        logger.debug("SetProcessDpiAwareness failed: %s", exc)

    try:
        user32.SetProcessDPIAware()
    except Exception as exc:
        logger.debug("SetProcessDPIAware failed: %s", exc)


class SingleInstanceMutex:
    """Named Windows mutex for single-instance enforcement."""

    def __init__(self, name: str = "Local\\KahootAnswerOverlay_Singleton_Mutex"):
        self.name = name
        self.handle = None
        self.already_running = False

    def acquire(self) -> bool:
        kernel32.SetLastError(0)
        handle = kernel32.CreateMutexW(None, False, self.name)
        err = kernel32.GetLastError()
        if not handle:
            logger.error("CreateMutexW failed with error %d", err)
            return True
        self.handle = handle
        if err == ERROR_ALREADY_EXISTS:
            self.already_running = True
            return False
        return True

    def release(self) -> None:
        if self.handle:
            try:
                kernel32.CloseHandle(self.handle)
            except Exception as exc:
                logger.debug("CloseHandle error: %s", exc)
            self.handle = None


def get_cursor_pos() -> tuple[int, int]:
    """Return current screen coordinates (x, y) of the mouse cursor."""
    pt = POINT()
    if user32.GetCursorPos(ctypes.byref(pt)):
        return int(pt.x), int(pt.y)
    return 0, 0


def _get_window_long_ptr(hwnd: int, index: int) -> int:
    if ctypes.sizeof(ctypes.c_void_p) == 8:
        fn = user32.GetWindowLongPtrW
        fn.restype = ctypes.c_longlong
        fn.argtypes = [wintypes.HWND, ctypes.c_int]
    else:
        fn = user32.GetWindowLongW
        fn.restype = ctypes.c_long
        fn.argtypes = [wintypes.HWND, ctypes.c_int]
    return int(fn(hwnd, index))


def _set_window_long_ptr(hwnd: int, index: int, new_long: int) -> int:
    if ctypes.sizeof(ctypes.c_void_p) == 8:
        fn = user32.SetWindowLongPtrW
        fn.restype = ctypes.c_longlong
        fn.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_longlong]
    else:
        fn = user32.SetWindowLongW
        fn.restype = ctypes.c_long
        fn.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_long]
    return int(fn(hwnd, index, new_long))


def apply_overlay_exstyles(hwnd: int, click_through: bool = False) -> None:
    """Apply WS_EX_TOOLWINDOW, WS_EX_NOACTIVATE, WS_EX_TOPMOST, and optional WS_EX_TRANSPARENT."""
    if not hwnd:
        return
    exstyle = _get_window_long_ptr(hwnd, GWL_EXSTYLE)
    exstyle |= WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE | WS_EX_LAYERED | WS_EX_TOPMOST
    if click_through:
        exstyle |= WS_EX_TRANSPARENT
    else:
        exstyle &= ~WS_EX_TRANSPARENT
    _set_window_long_ptr(hwnd, GWL_EXSTYLE, exstyle)
    user32.SetWindowPos(
        hwnd,
        HWND_TOPMOST,
        0,
        0,
        0,
        0,
        SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_FRAMECHANGED,
    )


def has_console() -> bool:
    """Return True if attached to a valid console with writable stdout."""
    try:
        hwnd = kernel32.GetConsoleWindow()
        if not hwnd:
            return False
        if sys.stdout is None or not hasattr(sys.stdout, "write"):
            return False
        return True
    except Exception:
        return False


def console_write(text: str) -> None:
    """Safely write text to stdout ONLY when launched from an attached console."""
    if has_console():
        try:
            sys.stdout.write(text + "\n")
            sys.stdout.flush()
        except Exception as exc:
            logger.debug("Console write error: %s", exc)


def open_file_in_os(path: str) -> None:
    """Open a file with the default Windows handler."""
    try:
        if os.path.exists(path):
            os.startfile(path)
    except Exception as exc:
        logger.error("Failed to open file %s: %s", path, exc)