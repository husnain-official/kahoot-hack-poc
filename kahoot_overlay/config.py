import json
import logging
import os
import tempfile
import threading
from typing import Any, Dict

logger = logging.getLogger("KahootOverlay.config")

DEFAULT_CONFIG: Dict[str, Any] = {
    "api_key": "",
    "model": "qwen/qwen3.8-27b",
    "reasoning_effort": "none",
    "text_model": "llama-3.3-70b-versatile",
    "pipeline": "single",
    "mode": "balanced",
    "tries": 3,
    "tries_temperature": 0.5,
    "early_exit": "majority",
    "timeout_ms": 2500,
    "latency_warn_ms": 1000,
    "image_max_edge": 1568,
    "jpeg_quality": 80,
    "keepalive_seconds": 20,
    "hotkeys": {
        "solve": "ctrl+alt+a",
        "toggle_overlay": "ctrl+alt+h",
        "quit": "ctrl+alt+q",
    },
    "overlay": {
        "position": "top-right",
        "width": 280,
        "height": 124,
        "opacity": 0.92,
        "click_through": False,
        "show_on_solve": True,
        "draggable": True,
        "auto_hide_seconds": 0.0,
    },
}


def get_app_dir() -> str:
    appdata = os.environ.get("APPDATA")
    if not appdata:
        appdata = os.path.expanduser("~")
    path = os.path.join(appdata, "KahootOverlay")
    os.makedirs(path, exist_ok=True)
    return path


def get_config_path() -> str:
    return os.path.join(get_app_dir(), "config.json")


def get_log_path() -> str:
    return os.path.join(get_app_dir(), "log.txt")


# Models Groq has shut down. A saved config naming one of these is migrated to the default.
RETIRED_MODELS = (
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "llama-3.2-11b-vision-preview",
    "llama-3.2-90b-vision-preview",
)


def _validate_and_merge(raw: Any) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if not isinstance(raw, dict):
        logger.warning("Config root is not a JSON object; using defaults.")
        return cfg

    for k in ("api_key", "model", "text_model"):
        if k in raw:
            if isinstance(raw[k], str) and (k == "api_key" or raw[k].strip()):
                cfg[k] = raw[k].strip()
            else:
                logger.warning("Invalid config value for %s; using default.", k)

    if cfg["model"] in RETIRED_MODELS:
        logger.warning("Configured model '%s' is retired; using '%s'.", cfg["model"], DEFAULT_CONFIG["model"])
        cfg["model"] = DEFAULT_CONFIG["model"]

    if "reasoning_effort" in raw:
        if raw["reasoning_effort"] in ("none", "low", "medium", "high"):
            cfg["reasoning_effort"] = raw["reasoning_effort"]
        else:
            logger.warning("Invalid reasoning_effort '%s'; using default.", raw["reasoning_effort"])

    if "pipeline" in raw:
        if raw["pipeline"] in ("single", "two_stage"):
            cfg["pipeline"] = raw["pipeline"]
        else:
            logger.warning("Invalid pipeline '%s'; using default.", raw["pipeline"])

    if "mode" in raw:
        if raw["mode"] in ("fast", "balanced"):
            cfg["mode"] = raw["mode"]
        else:
            logger.warning("Invalid mode '%s'; using default.", raw["mode"])

    if "tries" in raw:
        try:
            val = int(raw["tries"])
            if 1 <= val <= 9:
                cfg["tries"] = val
            else:
                logger.warning("tries=%s out of range 1..9; clamping.", val)
                cfg["tries"] = max(1, min(9, val))
        except (ValueError, TypeError):
            logger.warning("Invalid tries value; using default.")

    if "tries_temperature" in raw:
        try:
            val = float(raw["tries_temperature"])
            if 0.0 <= val <= 2.0:
                cfg["tries_temperature"] = val
            else:
                logger.warning("tries_temperature out of range; using default.")
        except (ValueError, TypeError):
            logger.warning("Invalid tries_temperature; using default.")

    if "early_exit" in raw:
        if raw["early_exit"] in ("majority", "off"):
            cfg["early_exit"] = raw["early_exit"]
        else:
            logger.warning("Invalid early_exit; using default.")

    int_bounds = {
        "timeout_ms": (300, 30000),
        "latency_warn_ms": (100, 30000),
        "image_max_edge": (512, 4096),
        "jpeg_quality": (20, 100),
        "keepalive_seconds": (0, 3600),
    }
    for k, (lo, hi) in int_bounds.items():
        if k in raw:
            try:
                val = int(raw[k])
                if lo <= val <= hi:
                    cfg[k] = val
                else:
                    logger.warning("%s=%s out of bounds [%d, %d]; using default.", k, val, lo, hi)
            except (ValueError, TypeError):
                logger.warning("Invalid %s; using default.", k)

    if "hotkeys" in raw and isinstance(raw["hotkeys"], dict):
        for hk_key in ("solve", "toggle_overlay", "quit"):
            val = raw["hotkeys"].get(hk_key)
            if isinstance(val, str) and val.strip():
                cfg["hotkeys"][hk_key] = val.strip().lower()
            elif val is not None:
                logger.warning("Invalid hotkey for %s; using default.", hk_key)

    if "overlay" in raw and isinstance(raw["overlay"], dict):
        ov = raw["overlay"]
        if "position" in ov:
            pos = ov["position"]
            if isinstance(pos, str) and pos.strip():
                cfg["overlay"]["position"] = pos.strip().lower()
            elif isinstance(pos, list) and len(pos) == 2 and all(isinstance(x, (int, float)) for x in pos):
                cfg["overlay"]["position"] = [int(pos[0]), int(pos[1])]
            else:
                logger.warning("Invalid overlay.position; using default.")

        for dim_key, lo, hi in (("width", 140, 1200), ("height", 60, 800)):
            if dim_key in ov:
                try:
                    val = int(ov[dim_key])
                    if lo <= val <= hi:
                        cfg["overlay"][dim_key] = val
                    else:
                        logger.warning("overlay.%s out of bounds; using default.", dim_key)
                except (ValueError, TypeError):
                    logger.warning("Invalid overlay.%s; using default.", dim_key)

        if "opacity" in ov:
            try:
                val = float(ov["opacity"])
                if 0.15 <= val <= 1.0:
                    cfg["overlay"]["opacity"] = val
                else:
                    logger.warning("overlay.opacity out of bounds; using default.")
            except (ValueError, TypeError):
                logger.warning("Invalid overlay.opacity; using default.")

        for bool_key in ("click_through", "show_on_solve", "draggable"):
            if bool_key in ov:
                if isinstance(ov[bool_key], bool):
                    cfg["overlay"][bool_key] = ov[bool_key]
                else:
                    logger.warning("Invalid overlay.%s; using default.", bool_key)

        if "auto_hide_seconds" in ov:
            try:
                val = float(ov["auto_hide_seconds"])
                if 0.0 <= val <= 3600.0:
                    cfg["overlay"]["auto_hide_seconds"] = val
                else:
                    logger.warning("overlay.auto_hide_seconds out of bounds; using default.")
            except (ValueError, TypeError):
                logger.warning("Invalid overlay.auto_hide_seconds; using default.")

    return cfg


class ConfigManager:
    """Thread-safe configuration manager with live reload and atomic writes."""

    def __init__(self, path: str | None = None):
        self.path = path or get_config_path()
        self._lock = threading.RLock()
        self._data: Dict[str, Any] = json.loads(json.dumps(DEFAULT_CONFIG))
        self._last_mtime: float = 0.0
        self.load(force=True)

    def load(self, force: bool = False) -> Dict[str, Any]:
        with self._lock:
            if not os.path.exists(self.path):
                logger.info("Config file missing at %s; creating with defaults.", self.path)
                self._data = json.loads(json.dumps(DEFAULT_CONFIG))
                self._atomic_write(self._data)
                try:
                    self._last_mtime = os.path.getmtime(self.path)
                except OSError:
                    self._last_mtime = 0.0
                return json.loads(json.dumps(self._data))

            try:
                mtime = os.path.getmtime(self.path)
            except OSError:
                mtime = 0.0

            if not force and mtime == self._last_mtime and mtime > 0:
                return json.loads(json.dumps(self._data))

            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                merged = _validate_and_merge(raw)
                self._data = merged
                self._last_mtime = mtime
                if raw != merged:
                    self._atomic_write(merged)
                    try:
                        self._last_mtime = os.path.getmtime(self.path)
                    except OSError:
                        pass
            except Exception as exc:
                logger.warning("Failed to parse %s (%s); falling back to defaults.", self.path, exc)
                self._data = json.loads(json.dumps(DEFAULT_CONFIG))
            return json.loads(json.dumps(self._data))

    def get(self) -> Dict[str, Any]:
        return self.load(force=False)

    def update(self, mutator_fn) -> Dict[str, Any]:
        with self._lock:
            current = self.load(force=False)
            mutator_fn(current)
            validated = _validate_and_merge(current)
            self._data = validated
            self._atomic_write(validated)
            try:
                self._last_mtime = os.path.getmtime(self.path)
            except OSError:
                pass
            return json.loads(json.dumps(self._data))

    def _atomic_write(self, data: Dict[str, Any]) -> None:
        folder = os.path.dirname(self.path)
        os.makedirs(folder, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix="cfg_", suffix=".tmp", dir=folder)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.path)
        except Exception as exc:
            logger.error("Atomic config write failed: %s", exc)
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass

    @staticmethod
    def _read_persisted_env(name: str) -> str:
        """Read a variable saved with setx / System Properties, even if this process
        was started from a shell that was opened before the variable existed."""
        try:
            import winreg

            locations = (
                (winreg.HKEY_CURRENT_USER, "Environment"),
                (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
            )
            for hive, sub in locations:
                try:
                    with winreg.OpenKey(hive, sub) as key:
                        value, _ = winreg.QueryValueEx(key, name)
                        if value and str(value).strip():
                            return str(value).strip()
                except OSError:
                    continue
        except Exception:
            pass
        return ""

    def get_effective_api_key(self) -> str:
        env_key = os.environ.get("GROQ_API_KEY", "").strip()
        if env_key:
            return env_key
        persisted = self._read_persisted_env("GROQ_API_KEY")
        if persisted:
            return persisted
        cfg = self.get()
        return str(cfg.get("api_key", "")).strip()