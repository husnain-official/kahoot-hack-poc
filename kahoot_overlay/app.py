import asyncio
import json
import logging
from logging.handlers import RotatingFileHandler
import queue
import statistics
import sys
import threading
import time
import tkinter as tk
from typing import Any, Dict, List, Optional

from capture import ScreenCapturer
from config import ConfigManager, get_config_path, get_log_path
from hotkeys import HotkeyListener
from overlay import OverlayWindow
from tray import TrayManager
from vision import VisionEngine, is_vision_model_name
from winutil import (
    SingleInstanceMutex,
    console_write,
     open_file_in_os,
    set_dpi_awareness,
)

logger = logging.getLogger("KahootOverlay")


def setup_logging() -> str:
    log_path = get_log_path()
    root_logger = logging.getLogger("KahootOverlay")
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    handler = RotatingFileHandler(
        log_path,
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.propagate = False
    return log_path


class KahootOverlayApp:
    """
    Main application coordinator wiring the Tkinter main thread, dedicated asyncio worker
    thread, Win32 RegisterHotKey listener thread, and pystray detached icon.
    """

    def __init__(self, config_mgr: ConfigManager, mutex: SingleInstanceMutex):
        self.config_mgr = config_mgr
        self.mutex = mutex
        self.ui_queue: "queue.Queue[tuple[str, Any]]" = queue.Queue()
        self.capturer = ScreenCapturer()

        self._shutting_down = False
        self._solve_seq = 0
        self._current_solve_task: Optional[asyncio.Future] = None
        self._solve_lock = threading.Lock()

        # Tkinter root on main thread
        self.root = tk.Tk()
        self.overlay = OverlayWindow(
            self.root,
            self.config_mgr.get(),
            on_position_changed=self._on_overlay_moved,
            monitor_at=self.capturer.get_monitor_at,
        )

        # Dedicated asyncio thread
        self.loop = asyncio.new_event_loop()
        self.vision = VisionEngine(
            notify_cb=self.enqueue_tray_notification,
            overlay_error_cb=self.enqueue_overlay_error,
        )
        self._async_thread = threading.Thread(
            target=self._run_async_loop,
            name="AsyncWorkerThread",
            daemon=True,
        )
        self._async_thread.start()

        # System tray
        self.tray = TrayManager(
            config_mgr=self.config_mgr,
            on_solve=lambda: self.ui_queue.put(("trigger_solve", time.perf_counter())),
            on_toggle_overlay=lambda: self.ui_queue.put(("toggle_overlay", None)),
            on_open_config=lambda: open_file_in_os(get_config_path()),
            on_open_log=lambda: open_file_in_os(get_log_path()),
            on_quit=lambda: self.ui_queue.put(("quit", None)),
            on_reset_overlay_position=lambda: self.ui_queue.put(("reset_overlay_position", None)),
            on_overlay_settings_changed=lambda: self.ui_queue.put(("apply_config", None)),
        )

        # Win32 global hotkeys
        cfg = self.config_mgr.get()
        self.hotkeys = HotkeyListener(
            hotkeys_cfg=cfg.get("hotkeys", {}),
            on_action=self._on_hotkey_action,
            on_registration_error=self._on_hotkey_reg_error,
        )

        self._install_exception_hooks()

    def _on_overlay_moved(self, offset_x: int, offset_y: int) -> None:
        """Called on the UI thread after the overlay is dragged; remember the spot."""
        self.config_mgr.update(
            lambda c: c.setdefault("overlay", {}).update({"position": [int(offset_x), int(offset_y)]})
        )
        logger.info("Overlay moved; saved offset (%d, %d).", offset_x, offset_y)

    def _install_exception_hooks(self) -> None:
        def handle_sys_exc(exc_type, exc_value, exc_tb):
            if issubclass(exc_type, KeyboardInterrupt):
                return
            logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))
            self.enqueue_overlay_error(f"{exc_type.__name__}")

        def handle_thread_exc(args):
            logger.error(
                "Uncaught thread exception in %s",
                getattr(args.thread, "name", "unknown"),
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            )
            self.enqueue_overlay_error(f"{args.exc_type.__name__}")

        sys.excepthook = handle_sys_exc
        threading.excepthook = handle_thread_exc

    def _run_async_loop(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def enqueue_tray_notification(self, title: str, message: str) -> None:
        self.ui_queue.put(("notify", (title, message)))

    def enqueue_overlay_error(self, short_reason: str) -> None:
        self.ui_queue.put(("overlay_error", short_reason))

    def _on_hotkey_action(self, action: str) -> None:
        t_press = time.perf_counter()
        if action == "solve":
            self.ui_queue.put(("trigger_solve", t_press))
        elif action == "toggle_overlay":
            self.ui_queue.put(("toggle_overlay", None))
        elif action == "quit":
            self.ui_queue.put(("quit", None))

    def _on_hotkey_reg_error(self, action: str, hk_str: str) -> None:
        self.enqueue_tray_notification(
            "Hotkey Conflict",
            f"Could not register '{hk_str}' for {action} (already in use).",
        )

    def start(self) -> None:
        self.tray.start()
        self.hotkeys.start()

        # Schedule startup warmup & keepalive loop on the async worker thread
        asyncio.run_coroutine_threadsafe(self._startup_and_keepalive_coro(), self.loop)

        # Start polling UI queue every 5 ms so hotkey -> Thinking... is well under 30 ms
        self.root.after(5, self._poll_ui_queue)
        logger.info("Entering Tkinter mainloop.")
        self.root.mainloop()

    async def _startup_and_keepalive_coro(self) -> None:
        cfg = self.config_mgr.get()
        api_key = self.config_mgr.get_effective_api_key()
        solve_hk = cfg.get("hotkeys", {}).get("solve", "ctrl+alt+a").upper()

        ok = await self.vision.warmup_and_verify_models(api_key, cfg)
        if ok:
            self.enqueue_tray_notification(
                "Kahoot Overlay",
                f"Kahoot Overlay ready · {solve_hk}",
            )

        while not self._shutting_down:
            cfg = self.config_mgr.get()
            interval = int(cfg.get("keepalive_seconds", 20))
            if interval <= 0:
                await asyncio.sleep(5.0)
                continue
            await asyncio.sleep(float(interval))
            if self._shutting_down:
                break
            key = self.config_mgr.get_effective_api_key()
            if key:
                await self.vision.keepalive_ping(key)

    def _poll_ui_queue(self) -> None:
        if self._shutting_down:
            return
        try:
            while True:
                cmd, payload = self.ui_queue.get_nowait()
                if cmd == "trigger_solve":
                    self._start_solve_on_main_thread(float(payload))
                elif cmd == "solve_result":
                    seq, result, timings, cfg, t_hotkey, t_thinking = payload
                    if seq == self._solve_seq:
                        t_render_start = time.perf_counter()
                        total_ms = (time.perf_counter() - t_hotkey) * 1000.0
                        self.overlay.show_result(result, total_ms=total_ms, cfg=cfg)
                        self.root.update_idletasks()
                        render_ms = (time.perf_counter() - t_render_start) * 1000.0
                        total_ms = (time.perf_counter() - t_hotkey) * 1000.0
                        timings["render_ms"] = render_ms
                        timings["total_ms"] = total_ms
                        self._log_solve_timings(seq, result, timings, cfg)
                elif cmd == "toggle_overlay":
                    self.overlay.toggle()
                elif cmd == "apply_config":
                    self.overlay.apply_config(self.config_mgr.get())
                elif cmd == "reset_overlay_position":
                    self.config_mgr.update(
                        lambda c: c.setdefault("overlay", {}).update({"position": "top-right"})
                    )
                    self.overlay.apply_config(self.config_mgr.get())
                    self.overlay.show()
                elif cmd == "overlay_error":
                    self.overlay.show_error(str(payload))
                elif cmd == "notify":
                    title, msg = payload
                    self.tray.notify(title, msg)
                elif cmd == "quit":
                    self.shutdown()
                    return
        except queue.Empty:
            pass
        except Exception as exc:
            logger.error("Error in _poll_ui_queue: %s", exc)

        if not self._shutting_down:
            self.root.after(5, self._poll_ui_queue)

    def _start_solve_on_main_thread(self, t_hotkey: float) -> None:
        """
        1. Reload config if changed.
        2. Hide overlay synchronously (< 2 ms) so it never contaminates the screenshot.
        3. Capture the monitor under the cursor.
        4. Restore overlay immediately in 'Thinking…' state.
        5. Cancel any previous in-flight solve task and start the new async solve.
        """
        cfg = self.config_mgr.get()
        self.overlay.apply_config(cfg)

        was_visible = self.overlay.hide_for_capture_sync()
        show_on_solve = bool(cfg.get("overlay", {}).get("show_on_solve", True))

        try:
            raw_shot, mon, capture_ms = self.capturer.grab_monitor_raw()
        except Exception as exc:
            logger.error("Screen capture failed (secure desktop or DRM?): %s", exc)
            self.overlay.restore_after_capture_thinking(was_visible, show_on_solve, None)
            self.overlay.show_error("Screen capture blocked")
            return

        self.overlay.restore_after_capture_thinking(was_visible, show_on_solve, mon)
        hotkey_to_thinking_ms = (time.perf_counter() - t_hotkey) * 1000.0

        with self._solve_lock:
            self._solve_seq += 1
            seq = self._solve_seq
            if self._current_solve_task is not None and not self._current_solve_task.done():
                self._current_solve_task.cancel()
                logger.info("Cancelled previous in-flight solve (latest press wins).")

            self._current_solve_task = asyncio.run_coroutine_threadsafe(
                self._async_solve_pipeline(
                    seq=seq,
                    raw_shot=raw_shot,
                    cfg=cfg,
                    t_hotkey=t_hotkey,
                    hotkey_to_thinking_ms=hotkey_to_thinking_ms,
                    capture_ms=capture_ms,
                ),
                self.loop,
            )

    async def _async_solve_pipeline(
        self,
        seq: int,
        raw_shot: Any,
        cfg: Dict[str, Any],
        t_hotkey: float,
        hotkey_to_thinking_ms: float,
        capture_ms: float,
    ) -> None:
        try:
            max_edge = int(cfg.get("image_max_edge", 1568))
            jpeg_q = int(cfg.get("jpeg_quality", 80))

            b64_img, encode_ms, dims = await asyncio.to_thread(
                self.capturer.encode_raw_shot,
                raw_shot,
                max_edge,
                jpeg_q,
            )

            api_key = self.config_mgr.get_effective_api_key()
            result, api_timings = await self.vision.solve_image(
                b64_image=b64_img,
                api_key=api_key,
                cfg=cfg,
            )

            timings = {
                "hotkey_to_thinking_ms": round(hotkey_to_thinking_ms, 1),
                "capture_ms": round(capture_ms, 1),
                "encode_ms": round(encode_ms, 1),
                "api_first_response_ms": round(api_timings.get("api_first_response_ms", 0.0), 1),
                "api_total_ms": round(api_timings.get("api_total_ms", 0.0), 1),
                "stage1_ms": round(api_timings.get("stage1_ms", 0.0), 1),
                "stage2_ms": round(api_timings.get("stage2_ms", 0.0), 1),
                "vote_ms": round(api_timings.get("vote_ms", 0.0), 1),
                "image_size": f"{dims[0]}x{dims[1]}",
            }
            self.ui_queue.put(("solve_result", (seq, result, timings, cfg, t_hotkey, hotkey_to_thinking_ms)))
        except asyncio.CancelledError:
            logger.debug("Solve #%d cancelled.", seq)
            raise
        except Exception as exc:
            logger.error("Unexpected error in solve #%d: %s", seq, exc)
            err_res = {"status": "error", "error": f"Solve error: {type(exc).__name__}", "found": False}
            self.ui_queue.put(("solve_result", (seq, err_res, {}, cfg, t_hotkey, hotkey_to_thinking_ms)))

    def _log_solve_timings(
        self,
        seq: int,
        result: Dict[str, Any],
        timings: Dict[str, Any],
        cfg: Dict[str, Any],
    ) -> None:
        total_ms = float(timings.get("total_ms", 0.0))
        warn_ms = float(cfg.get("latency_warn_ms", 1000))
        breakdown = (
            f"solve=#{seq} status={result.get('status')} answer={result.get('answer')} "
            f"conf={result.get('confidence')}% | "
            f"hotkey->thinking={timings.get('hotkey_to_thinking_ms', 0):.1f}ms "
            f"capture={timings.get('capture_ms', 0):.1f}ms "
            f"encode={timings.get('encode_ms', 0):.1f}ms "
            f"api_first={timings.get('api_first_response_ms', 0):.1f}ms "
            f"api_total={timings.get('api_total_ms', 0):.1f}ms "
            f"vote={timings.get('vote_ms', 0):.1f}ms "
            f"render={timings.get('render_ms', 0):.1f}ms "
            f"total={total_ms:.1f}ms"
        )
        if total_ms > warn_ms:
            logger.warning("LATENCY BUDGET EXCEEDED (%.1fms > %.0fms): %s", total_ms, warn_ms, breakdown)
        else:
            logger.info("Solve complete: %s", breakdown)

    def shutdown(self) -> None:
        if self._shutting_down:
            return
        self._shutting_down = True
        logger.info("Shutting down Kahoot Answer Overlay...")

        try:
            self.hotkeys.stop()
        except Exception as exc:
            logger.debug("Hotkeys stop error: %s", exc)

        try:
            self.tray.stop()
        except Exception as exc:
            logger.debug("Tray stop error: %s", exc)

        try:
            if self._current_solve_task and not self._current_solve_task.done():
                self._current_solve_task.cancel()
            fut = asyncio.run_coroutine_threadsafe(self.vision.close(), self.loop)
            fut.result(timeout=1.5)
        except Exception as exc:
            logger.debug("Vision client close error: %s", exc)

        try:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self._async_thread.join(timeout=1.5)
        except Exception as exc:
            logger.debug("Async loop stop error: %s", exc)

        try:
            self.capturer.close()
        except Exception:
            pass

        try:
            self.mutex.release()
        except Exception:
            pass

        try:
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass
        logger.info("Clean shutdown complete.")


# ---------------------------------------------------------------------------
# Headless CLI Modes (--selftest, --bench, --list-models)
# ---------------------------------------------------------------------------

async def _run_cli_selftest(image_path: str, config_mgr: ConfigManager) -> int:
    cfg = config_mgr.get()
    api_key = config_mgr.get_effective_api_key()
    capturer = ScreenCapturer()
    engine = VisionEngine()
    try:
        t0 = time.perf_counter()
        await engine.warmup_and_verify_models(api_key, cfg)
        t_solve_start = time.perf_counter()
        b64_img, capture_ms, encode_ms, dims = capturer.load_file_and_encode(
            image_path,
            max_edge=int(cfg.get("image_max_edge", 1568)),
            jpeg_quality=int(cfg.get("jpeg_quality", 80)),
        )
        result, api_timings = await engine.solve_image(b64_img, api_key, cfg)
        total_ms = (time.perf_counter() - t_solve_start) * 1000.0

        stage_table = {
            "hotkey_to_thinking_ms": 0.0,
            "capture_ms": round(capture_ms, 2),
            "encode_ms": round(encode_ms, 2),
            "api_first_response_ms": round(api_timings.get("api_first_response_ms", 0.0), 2),
            "api_total_ms": round(api_timings.get("api_total_ms", 0.0), 2),
            "vote_ms": round(api_timings.get("vote_ms", 0.0), 2),
            "render_ms": 0.0,
            "total_ms": round(total_ms, 2),
            "image_dims": f"{dims[0]}x{dims[1]}",
        }

        out_payload = {"result": result, "timings_ms": stage_table}
        formatted = json.dumps(out_payload, indent=2)
        logger.info("SELFTEST output:\n%s", formatted)
        console_write(formatted)
        return 0 if result.get("status") in ("ok", "no_question") else 1
    finally:
        await engine.close()


async def _run_cli_bench(runs: int, image_path: str, config_mgr: ConfigManager) -> int:
    cfg = config_mgr.get()
    api_key = config_mgr.get_effective_api_key()
    capturer = ScreenCapturer()
    engine = VisionEngine()
    try:
        await engine.warmup_and_verify_models(api_key, cfg)
        latencies: List[float] = []
        results: List[Dict[str, Any]] = []

        for i in range(runs):
            t0 = time.perf_counter()
            b64_img, capture_ms, encode_ms, _ = capturer.load_file_and_encode(
                image_path,
                max_edge=int(cfg.get("image_max_edge", 1568)),
                jpeg_quality=int(cfg.get("jpeg_quality", 80)),
            )
            res, api_timings = await engine.solve_image(b64_img, api_key, cfg)
            total_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(total_ms)
            results.append(res)
            logger.info("Bench run %d/%d: %.1f ms -> %s", i + 1, runs, total_ms, res)

        latencies_sorted = sorted(latencies)
        p50 = statistics.median(latencies_sorted) if latencies_sorted else 0.0
        if len(latencies_sorted) >= 2:
            idx95 = min(len(latencies_sorted) - 1, int(round(0.95 * (len(latencies_sorted) - 1))))
            p95 = latencies_sorted[idx95]
        else:
            p95 = p50

        summary = {
            "runs": runs,
            "p50_ms": round(p50, 2),
            "p95_ms": round(p95, 2),
            "min_ms": round(min(latencies_sorted), 2) if latencies_sorted else 0.0,
            "max_ms": round(max(latencies_sorted), 2) if latencies_sorted else 0.0,
            "all_ms": [round(x, 1) for x in latencies],
            "last_result": results[-1] if results else None,
        }
        text = json.dumps(summary, indent=2)
        logger.info("BENCH summary:\n%s", text)
        console_write(text)
        return 0
    finally:
        await engine.close()


async def _run_cli_list_models(config_mgr: ConfigManager) -> int:
    api_key = config_mgr.get_effective_api_key()
    engine = VisionEngine()
    try:
        models = await engine.list_models(api_key)
        flagged = [
            {"id": m, "vision_capable": is_vision_model_name(m)}
            for m in models
        ]
        text = json.dumps({"models": flagged}, indent=2)
        logger.info("LIST-MODELS:\n%s", text)
        console_write(text)
        return 0
    except Exception as exc:
        err = json.dumps({"error": str(exc)})
        logger.error("LIST-MODELS failed: %s", exc)
        console_write(err)
        return 1
    finally:
        await engine.close()


def run_application(argv: Optional[List[str]] = None) -> int:
    # M7: Set Per-Monitor DPI awareness BEFORE creating any window or querying monitors
    set_dpi_awareness()
    setup_logging()
    config_mgr = ConfigManager()

    args = list(sys.argv[1:] if argv is None else argv)

    # Headless CLI flags (work without tray or hotkeys)
    if args:
        if args[0] == "--selftest" and len(args) >= 2:
            return asyncio.run(_run_cli_selftest(args[1], config_mgr))
        if args[0] == "--bench" and len(args) >= 3:
            try:
                n_runs = max(1, int(args[1]))
            except ValueError:
                n_runs = 1
            return asyncio.run(_run_cli_bench(n_runs, args[2], config_mgr))
        if args[0] == "--list-models":
            return asyncio.run(_run_cli_list_models(config_mgr))

    # Normal GUI mode: enforce single instance via named Win32 mutex
    mutex = SingleInstanceMutex()
    if not mutex.acquire():
        logger.info("Another instance of KahootOverlay is already running; exiting.")
        return 0

    app = KahootOverlayApp(config_mgr=config_mgr, mutex=mutex)
    app.start()
    return 0