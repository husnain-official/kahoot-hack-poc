"""Entry point. Launch with: pythonw main.pyw  (or double-click).

Any startup failure is written to %APPDATA%\\KahootOverlay\\crash.txt and shown in a
message box, because under pythonw there is no console to print errors to.
"""
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _crash_file() -> str:
    base = os.environ.get("APPDATA") or os.path.expanduser("~")
    folder = os.path.join(base, "KahootOverlay")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, "crash.txt")


def _report_fatal(details: str) -> None:
    path = ""
    try:
        path = _crash_file()
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n" + details + "\n" + "-" * 60 + "\n")
    except Exception:
        pass
    try:
        import ctypes

        last_line = details.strip().splitlines()[-1] if details.strip() else "unknown error"
        ctypes.windll.user32.MessageBoxW(
            0,
            f"Kahoot Overlay failed to start:\n\n{last_line}\n\nFull details:\n{path}",
            "Kahoot Overlay",
            0x10,  # MB_ICONERROR
        )
    except Exception:
        pass


def main() -> int:
    try:
        from app import run_application

        return int(run_application() or 0)
    except SystemExit:
        raise
    except BaseException:
        _report_fatal(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
