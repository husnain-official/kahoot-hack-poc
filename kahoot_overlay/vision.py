import asyncio
import json
import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import groq
from groq import AsyncGroq

from voting import parse_model_response, parse_stage1_extraction, vote

logger = logging.getLogger("KahootOverlay.vision")

SYSTEM_PROMPT = """You are a Kahoot answer engine. The image is a screenshot of a monitor. Somewhere in it is a Kahoot question screen (it may be a small browser window among other apps). Ignore everything that is not the Kahoot question and its answer tiles.

Kahoot answer tiles are identified by COLOR and SHAPE: red triangle, blue diamond, yellow circle, green square. In the standard 2x2 layout they are red=top-left, blue=top-right, yellow=bottom-left, green=bottom-right, but report the position you actually see on screen. Some questions have only 2 tiles.

Rules:
1. Read the question and the text of every visible tile. Tiles may contain only an image or shape; use the question and any visible image to decide.
2. Choose the correct tile(s).
3. For True/False questions, find the tile whose printed label matches your answer and report THAT tile's color. Never assume which color means True or False.
4. If the question asks to select multiple answers, list all correct tiles. If it is an ordering/puzzle question, list tiles in the correct order.
5. If the screen shows no active Kahoot question (lobby, results, scoreboard, other app), set found=false.
6. Reply with a single JSON object and nothing else.

conf is an integer 0-100 for how likely your chosen answer is correct: 90-100 you read everything clearly AND know the fact; 60-89 likely but some doubt; 30-59 educated guess; below 30 text unreadable or you do not know. Do not default to high numbers."""

BALANCED_SCHEMA_INSTRUCTION = (
    'Return ONLY a JSON object with keys in this exact order: '
    '{"found": true, "type": "single|multi|truefalse|order", "think": "<=30 words of reasoning", '
    '"answer": ["red"], "pos": ["top-left"], "conf": 85}. '
    'Valid answer colors: "red", "blue", "yellow", "green". '
    'Valid pos values: "top-left", "top-right", "bottom-left", "bottom-right", "left", "right", "top", "bottom", "center".'
)

FAST_SCHEMA_INSTRUCTION = (
    'Return ONLY a compact JSON object without any "think" field: '
    '{"found": true, "type": "single|multi|truefalse|order", '
    '"answer": ["red"], "pos": ["top-left"], "conf": 85}. '
    'Valid answer colors: "red", "blue", "yellow", "green". '
    'Valid pos values: "top-left", "top-right", "bottom-left", "bottom-right", "left", "right", "top", "bottom", "center".'
)

STAGE1_EXTRACTION_PROMPT = """You are a fast OCR extractor for Kahoot screens.
Read the monitor screenshot and extract the visible Kahoot question and answer tiles.
For True/False questions, copy the exact printed word ("True" or "False") on each tile color. Never guess or assume colors.
If no active Kahoot question with answer tiles is shown, return {"found": false}.
Otherwise return ONLY a compact JSON object:
{"found": true, "type": "single|multi|truefalse|order", "question": "...", "options": {"red": "...", "blue": "...", "yellow": "...", "green": "..."}, "positions": {"red": "top-left", "blue": "top-right", "yellow": "bottom-left", "green": "bottom-right"}}"""

STAGE2_SYSTEM_PROMPT = """You are a Kahoot trivia solver. You are given the extracted question text, tile labels by color, and tile positions on screen.
Rules:
1. Choose the correct tile color(s) based on the text of each tile.
2. For True/False questions, check which tile color has the label matching your answer. Never assume which color means True or False.
3. Reply with a single JSON object and nothing else.
conf is an integer 0-100 for how likely your answer is correct (90-100 certain, 60-89 likely, 30-59 guess, <30 unknown)."""

# Slight prompt phrasing variations for tries #2..N so independent tries have distinct context
# in addition to higher temperature and distinct random seeds.
TRY_PERSPECTIVE_HINTS = [
    "Verify the question wording and each tile label carefully.",
    "Double-check dates, numbers, and negation words (like NOT or NEVER) in the question before picking the tile.",
    "Inspect every option from bottom-right to top-left to avoid anchoring bias, then choose the best tile.",
    "Cross-check factual details and eliminate clearly wrong tiles first, then report the winning tile color.",
    "Read the exact printed text inside each colored shape carefully before deciding.",
    "Confirm whether the question requires a single tile, multiple tiles, or True/False tile label match.",
    "Evaluate each tile option independently against the question prompt.",
    "Verify there are no trick words in the question or tile labels, then output your JSON verdict.",
    "Check each colored tile's text and on-screen position before finalizing your JSON response.",
]


# Newest first. Groq retired Llama 4 Scout/Maverick (2026); Qwen 3.x are the current vision models.
PREFERRED_VISION_MODELS = ("qwen/qwen3.8-27b", "qwen/qwen3.6-27b")

_QWEN_VISION_RE = re.compile(r"qwen3\.[5-9]|qwen3-vl|qwen2\.5-vl|qwen-vl")
_NON_CHAT_HINTS = ("whisper", "tts", "guard", "embed", "orpheus", "moderation")


def is_vision_model_name(model_id: str) -> bool:
    m = model_id.lower()
    if any(h in m for h in _NON_CHAT_HINTS):
        return False
    if m.startswith("llama-3.2-") and "vision" in m:
        return False  # retired previews
    if _QWEN_VISION_RE.search(m):
        return True
    return any(k in m for k in ("scout", "maverick", "vision", "llava", "pixtral"))


def select_best_vision_model(available_ids: List[str], configured_model: str) -> Optional[str]:
    if configured_model in available_ids:
        return configured_model

    for preferred in PREFERRED_VISION_MODELS:
        if preferred in available_ids:
            return preferred

    candidates = [m for m in available_ids if is_vision_model_name(m)]
    if not candidates:
        return None

    qwen = sorted((m for m in candidates if "qwen" in m.lower()), reverse=True)
    if qwen:
        return qwen[0]

    for keyword in ("maverick", "scout", "vision"):
        matches = sorted(m for m in candidates if keyword in m.lower())
        if matches:
            return matches[0]

    return sorted(candidates)[0]


def select_best_text_model(available_ids: List[str], configured_text_model: str) -> Optional[str]:
    if configured_text_model in available_ids:
        return configured_text_model
    for keyword in ("llama-3.3-70b", "llama-3.1-70b", "qwen", "llama", "mixtral"):
        matches = [m for m in available_ids if keyword in m.lower() and "whisper" not in m.lower() and "tts" not in m.lower()]
        if matches:
            return matches[0]
    return available_ids[0] if available_ids else configured_text_model


class VisionEngine:
    """
    Manages the persistent AsyncGroq client, model discovery, keep-alive pings,
    parallel tries with majority early exit, and 429 rate-limit backoff.
    Must be driven from a single dedicated asyncio event loop.
    """

    def __init__(
        self,
        notify_cb: Optional[Callable[[str, str], None]] = None,
        overlay_error_cb: Optional[Callable[[str], None]] = None,
    ):
        self._client: Optional[AsyncGroq] = None
        self._client_api_key: str = ""
        self._notify_cb = notify_cb
        self._overlay_error_cb = overlay_error_cb

        self.active_vision_model: Optional[str] = None
        self.active_text_model: Optional[str] = None
        self.available_models: List[str] = []
        self._models_verified: bool = False

        # Session memory for whether a given model supports response_format={"type": "json_object"}
        self._json_mode_supported: Dict[str, bool] = {}
        # Reasoning control (Qwen 3.x on Groq): "none" turns thinking off for speed.
        self._reasoning_effort: str = "none"
        self._reasoning_supported: Dict[str, bool] = {}

        # Session rate-limit cap (reduced on HTTP 429)
        self.session_max_tries: Optional[int] = None

    def ensure_client(self, api_key: str) -> Optional[AsyncGroq]:
        if not api_key:
            return None
        if self._client is None or self._client_api_key != api_key:
            if self._client is not None:
                try:
                    asyncio.create_task(self._client.close())
                except Exception:
                    pass
            self._client = AsyncGroq(api_key=api_key, max_retries=0)
            self._client_api_key = api_key
            self._models_verified = False
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            try:
                await self._client.close()
            except Exception as exc:
                logger.debug("Error closing AsyncGroq client: %s", exc)
            self._client = None

    async def list_models(self, api_key: str) -> List[str]:
        client = self.ensure_client(api_key)
        if client is None:
            raise RuntimeError("Missing GROQ_API_KEY")
        resp = await client.models.list(timeout=5.0)
        ids = [m.id for m in getattr(resp, "data", []) if getattr(m, "id", None)]
        self.available_models = sorted(ids)
        return self.available_models

    async def warmup_and_verify_models(self, api_key: str, cfg: Dict[str, Any]) -> bool:
        """Warm up TLS connection and verify configured models against Groq's models endpoint."""
        if not api_key:
            msg = "No API key found. Tray menu > Open config file, paste it into api_key, restart."
            logger.warning(msg)
            if self._overlay_error_cb:
                self._overlay_error_cb(msg)
            if self._notify_cb:
                self._notify_cb("Kahoot Overlay: Missing API Key", msg)
            return False

        try:
            ids = await self.list_models(api_key)
        except groq.AuthenticationError:
            msg = "Invalid Groq API Key"
            logger.error(msg)
            if self._overlay_error_cb:
                self._overlay_error_cb(msg)
            if self._notify_cb:
                self._notify_cb("Kahoot Overlay Error", msg)
            return False
        except Exception as exc:
            logger.warning("Startup model list warmup failed: %s", exc)
            return False

        self._reasoning_effort = str(cfg.get("reasoning_effort", "none")).lower()
        configured_vision = cfg.get("model", "qwen/qwen3.8-27b")
        chosen_vision = select_best_vision_model(ids, configured_vision)
        if not chosen_vision:
            msg = "No vision-capable model available on Groq account"
            logger.error(msg)
            if self._overlay_error_cb:
                self._overlay_error_cb(msg)
            if self._notify_cb:
                self._notify_cb("Kahoot Overlay Error", msg)
            return False

        if chosen_vision != configured_vision:
            logger.warning(
                "Configured model '%s' not found; substituted '%s'.",
                configured_vision,
                chosen_vision,
            )
            if self._notify_cb:
                self._notify_cb(
                    "Kahoot Overlay: Model Substituted",
                    f"Using {chosen_vision}",
                )
        self.active_vision_model = chosen_vision

        configured_text = cfg.get("text_model", "llama-3.3-70b-versatile")
        self.active_text_model = select_best_text_model(ids, configured_text)
        self._models_verified = True
        logger.info(
            "Warmup complete. active_vision_model=%s, active_text_model=%s",
            self.active_vision_model,
            self.active_text_model,
        )
        return True

    async def keepalive_ping(self, api_key: str) -> None:
        """Cheap 0-token keep-alive call to keep the HTTP/2 or TLS session warm."""
        if not api_key:
            return
        client = self.ensure_client(api_key)
        if client is None:
            return
        try:
            await client.models.list(timeout=4.0)
            logger.debug("Keep-alive ping succeeded.")
        except Exception as exc:
            logger.debug("Keep-alive ping failed: %s", exc)

    _REASONING_HEADROOM = {"none": 0, "low": 384, "medium": 1024, "high": 2048}

    async def _call_chat_completion(
        self,
        client: AsyncGroq,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
        seed: Optional[int],
        timeout_sec: float,
    ) -> str:
        use_json_fmt = self._json_mode_supported.get(model, True)
        effort = self._reasoning_effort if "qwen" in model.lower() else ""
        use_reasoning = bool(effort) and self._reasoning_supported.get(model, True)

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout_sec,
        }
        if seed is not None:
            kwargs["seed"] = seed
        if use_json_fmt:
            kwargs["response_format"] = {"type": "json_object"}
        if use_reasoning:
            # Passed via extra_body so it works on every groq SDK version.
            extra: Dict[str, Any] = {"reasoning_effort": effort}
            if effort != "none":
                extra["reasoning_format"] = "hidden"  # keep the reply pure JSON
                kwargs["max_tokens"] = max_tokens + self._REASONING_HEADROOM.get(effort, 1024)
            kwargs["extra_body"] = extra

        # Each pass removes at most one unsupported parameter, then retries.
        for _ in range(4):
            try:
                resp = await client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except groq.BadRequestError as exc:
                err_body = str(exc).lower()
                if "extra_body" in kwargs and "reasoning" in err_body:
                    logger.info("Model %s rejected reasoning parameters; retrying without them.", model)
                    self._reasoning_supported[model] = False
                    kwargs.pop("extra_body", None)
                    kwargs["max_tokens"] = max_tokens
                    continue
                if "response_format" in kwargs and ("response_format" in err_body or "json" in err_body):
                    logger.info("Model %s rejected response_format=json_object; falling back to plain text.", model)
                    self._json_mode_supported[model] = False
                    kwargs.pop("response_format", None)
                    continue
                if "seed" in kwargs and "seed" in err_body:
                    kwargs.pop("seed", None)
                    continue
                raise
        raise RuntimeError("Request rejected after removing unsupported parameters")

    async def _single_vision_try(
        self,
        client: AsyncGroq,
        model: str,
        b64_image: str,
        mode: str,
        try_index: int,
        tries_temperature: float,
        timeout_sec: float,
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        t0 = time.perf_counter()
        is_temp_zero = try_index == 0
        temp = 0.0 if is_temp_zero else float(tries_temperature)
        seed = None if is_temp_zero else (1000 + try_index * 997 + int(time.time() * 1000) % 10000)
        max_tokens = 160 if mode == "balanced" else 60
        schema_instr = BALANCED_SCHEMA_INSTRUCTION if mode == "balanced" else FAST_SCHEMA_INSTRUCTION
        hint = TRY_PERSPECTIVE_HINTS[try_index % len(TRY_PERSPECTIVE_HINTS)]
        user_text = f"{hint}\n{schema_instr}"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                    },
                ],
            },
        ]

        raw_text = await self._call_chat_completion(
            client=client,
            model=model,
            messages=messages,
            temperature=temp,
            max_tokens=max_tokens,
            seed=seed,
            timeout_sec=timeout_sec,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        parsed = parse_model_response(raw_text, try_index=try_index, is_temp_zero=is_temp_zero)
        if parsed is None:
            logger.debug("Try #%d returned unparseable output: %r", try_index + 1, raw_text[:160])
        return parsed, elapsed_ms

    async def _stage2_text_try(
        self,
        client: AsyncGroq,
        text_model: str,
        extracted: Dict[str, Any],
        mode: str,
        try_index: int,
        tries_temperature: float,
        timeout_sec: float,
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        t0 = time.perf_counter()
        is_temp_zero = try_index == 0
        temp = 0.0 if is_temp_zero else float(tries_temperature)
        seed = None if is_temp_zero else (2000 + try_index * 997 + int(time.time() * 1000) % 10000)
        max_tokens = 160 if mode == "balanced" else 60
        schema_instr = BALANCED_SCHEMA_INSTRUCTION if mode == "balanced" else FAST_SCHEMA_INSTRUCTION
        hint = TRY_PERSPECTIVE_HINTS[try_index % len(TRY_PERSPECTIVE_HINTS)]

        payload_desc = json.dumps(
            {
                "type": extracted.get("type", "single"),
                "question": extracted.get("question", ""),
                "options": extracted.get("options", {}),
                "positions": extracted.get("positions", {}),
            },
            ensure_ascii=False,
        )
        user_text = f"Extracted Kahoot screen:\n{payload_desc}\n\n{hint}\n{schema_instr}"

        messages = [
            {"role": "system", "content": STAGE2_SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]

        raw_text = await self._call_chat_completion(
            client=client,
            model=text_model,
            messages=messages,
            temperature=temp,
            max_tokens=max_tokens,
            seed=seed,
            timeout_sec=timeout_sec,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        parsed = parse_model_response(raw_text, try_index=try_index, is_temp_zero=is_temp_zero)
        return parsed, elapsed_ms

    async def _run_parallel_tries(
        self,
        coro_factory: Callable[[int, float], Any],
        effective_tries: int,
        early_exit_mode: str,
        timeout_sec: float,
    ) -> Tuple[List[Dict[str, Any]], bool, float, Optional[str], bool]:
        """
        Launch N tries concurrently, implement majority early exit, and handle HTTP 429 / timeouts.
        Returns (valid_tries, early_exited, first_response_ms, last_error_reason, hit_rate_limit).
        """
        t_start = time.perf_counter()
        tasks: List[asyncio.Task] = [
            asyncio.create_task(coro_factory(i, timeout_sec)) for i in range(effective_tries)
        ]

        valid_tries: List[Dict[str, Any]] = []
        early_exited = False
        first_response_ms = 0.0
        last_error: Optional[str] = None
        hit_429 = False

        majority_threshold = (effective_tries // 2) + 1
        use_early_exit = early_exit_mode == "majority" and effective_tries > 1

        try:
            for fut in asyncio.as_completed(tasks):
                try:
                    parsed, try_ms = await fut
                    if first_response_ms == 0.0:
                        first_response_ms = try_ms
                    if parsed is not None:
                        valid_tries.append(parsed)
                        if use_early_exit:
                            if self._has_majority(valid_tries, majority_threshold, effective_tries):
                                early_exited = len(valid_tries) < effective_tries
                                break
                except asyncio.CancelledError:
                    raise
                except groq.RateLimitError:
                    hit_429 = True
                    last_error = "Rate limited (HTTP 429)"
                    logger.warning("Try hit HTTP 429 RateLimitError.")
                except groq.AuthenticationError:
                    last_error = "Invalid API key"
                    break
                except groq.NotFoundError as exc:
                    last_error = "Model not found on your Groq account. Check \"model\" in config.json."
                    self._models_verified = False
                    logger.error("Model not found (404): %s", exc)
                    break
                except groq.APITimeoutError:
                    last_error = "API request timed out"
                    logger.warning("Try timed out after %.0f ms", timeout_sec * 1000)
                except groq.APIConnectionError:
                    last_error = "Network connection error"
                    logger.warning("APIConnectionError during try.")
                except Exception as exc:
                    last_error = f"API error ({type(exc).__name__})"
                    logger.warning("Try failed with %s: %s", type(exc).__name__, exc)
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        # If all tries failed due to 429 and time budget remains, retry a single try once
        elapsed_sec = time.perf_counter() - t_start
        remaining_sec = timeout_sec - elapsed_sec
        if not valid_tries and hit_429 and remaining_sec > 0.4:
            logger.info("Retrying once with 1 try after HTTP 429 (%.2fs budget left).", remaining_sec)
            try:
                parsed, try_ms = await coro_factory(0, remaining_sec)
                if parsed is not None:
                    valid_tries.append(parsed)
                    if first_response_ms == 0.0:
                        first_response_ms = try_ms
            except Exception as exc:
                logger.warning("Single retry after 429 failed: %s", exc)

        return valid_tries, early_exited, first_response_ms, last_error, hit_429

    @staticmethod
    def _has_majority(valid_tries: List[Dict[str, Any]], majority_threshold: int, effective_tries: int) -> bool:
        if effective_tries <= 2:
            # Per Section 7.3: "With tries = 1 or 2, early exit simply means 'first valid answer(s)'."
            return len(valid_tries) >= 1

        not_found_count = sum(1 for t in valid_tries if not t.get("found", False))
        if not_found_count >= majority_threshold:
            return True

        counts: Dict[Tuple[str, ...], int] = {}
        for t in valid_tries:
            if t.get("found", False) and t.get("answer"):
                key = tuple(t["answer"])
                counts[key] = counts.get(key, 0) + 1
                if counts[key] >= majority_threshold:
                    return True
        return False

    async def solve_image(
        self,
        b64_image: str,
        api_key: str,
        cfg: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Execute single-stage or two-stage vision + voting pipeline on a base64 JPEG.
        Returns (result_dict, timings_dict).
        """
        timings: Dict[str, float] = {
            "api_first_response_ms": 0.0,
            "api_total_ms": 0.0,
            "stage1_ms": 0.0,
            "stage2_ms": 0.0,
            "vote_ms": 0.0,
        }

        if not api_key:
            return {
                "status": "error",
                "error": "No API key found. Tray menu > Open config file, paste it into api_key, restart.",
                "found": False,
            }, timings

        client = self.ensure_client(api_key)
        if client is None:
            return {"status": "error", "error": "API client unavailable", "found": False}, timings

        self._reasoning_effort = str(cfg.get("reasoning_effort", "none")).lower()

        if not self._models_verified:
            asyncio.create_task(self.warmup_and_verify_models(api_key, cfg))

        vision_model = self.active_vision_model or cfg.get("model", "qwen/qwen3.8-27b")
        text_model = self.active_text_model or cfg.get("text_model", "llama-3.3-70b-versatile")
        pipeline = cfg.get("pipeline", "single")
        mode = cfg.get("mode", "balanced")
        configured_tries = int(cfg.get("tries", 3))
        tries_temp = float(cfg.get("tries_temperature", 0.5))
        early_exit_mode = cfg.get("early_exit", "majority")
        timeout_sec = max(0.3, float(cfg.get("timeout_ms", 2500)) / 1000.0)

        effective_tries = configured_tries
        if self.session_max_tries is not None:
            effective_tries = min(effective_tries, self.session_max_tries)

        t_api_start = time.perf_counter()

        if pipeline == "two_stage":
            # Stage 1: Vision OCR extraction
            t_s1 = time.perf_counter()
            s1_messages = [
                {"role": "system", "content": STAGE1_EXTRACTION_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract the Kahoot question and tile labels as JSON."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
                    ],
                },
            ]
            try:
                s1_raw = await self._call_chat_completion(
                    client=client,
                    model=vision_model,
                    messages=s1_messages,
                    temperature=0.0,
                    max_tokens=200,
                    seed=None,
                    timeout_sec=timeout_sec,
                )
            except groq.AuthenticationError:
                return {"status": "error", "error": "Invalid API key", "found": False}, timings
            except groq.NotFoundError:
                self._models_verified = False
                return {"status": "error", "error": "Model not found on your Groq account. Check \"model\" in config.json.", "found": False}, timings
            except groq.RateLimitError:
                self.session_max_tries = max(1, effective_tries // 2)
                return {
                    "status": "error",
                    "error": f"Rate limited, using {self.session_max_tries} tries",
                    "found": False,
                }, timings
            except groq.APITimeoutError:
                return {"status": "error", "error": "Stage 1 vision timed out", "found": False}, timings
            except groq.APIConnectionError:
                return {"status": "error", "error": "No network connection", "found": False}, timings
            except Exception as exc:
                return {"status": "error", "error": f"Vision error: {type(exc).__name__}", "found": False}, timings

            timings["stage1_ms"] = (time.perf_counter() - t_s1) * 1000.0
            extracted = parse_stage1_extraction(s1_raw)
            if extracted is None:
                return {"status": "error", "error": "Malformed Stage 1 JSON", "found": False}, timings
            if not extracted.get("found", False):
                timings["api_total_ms"] = (time.perf_counter() - t_api_start) * 1000.0
                timings["api_first_response_ms"] = timings["stage1_ms"]
                return {"status": "no_question", "found": False}, timings

            # Stage 2: Parallel text model tries
            t_s2 = time.perf_counter()
            valid_tries, early_exited, first_ms, last_err, hit_429 = await self._run_parallel_tries(
                coro_factory=lambda idx, t_sec: self._stage2_text_try(
                    client=client,
                    text_model=text_model,
                    extracted=extracted,
                    mode=mode,
                    try_index=idx,
                    tries_temperature=tries_temp,
                    timeout_sec=t_sec,
                ),
                effective_tries=effective_tries,
                early_exit_mode=early_exit_mode,
                timeout_sec=timeout_sec,
            )
            timings["stage2_ms"] = (time.perf_counter() - t_s2) * 1000.0
            timings["api_first_response_ms"] = timings["stage1_ms"] + first_ms
        else:
            # Single-stage parallel vision tries
            valid_tries, early_exited, first_ms, last_err, hit_429 = await self._run_parallel_tries(
                coro_factory=lambda idx, t_sec: self._single_vision_try(
                    client=client,
                    model=vision_model,
                    b64_image=b64_image,
                    mode=mode,
                    try_index=idx,
                    tries_temperature=tries_temp,
                    timeout_sec=t_sec,
                ),
                effective_tries=effective_tries,
                early_exit_mode=early_exit_mode,
                timeout_sec=timeout_sec,
            )
            timings["api_first_response_ms"] = first_ms

        timings["api_total_ms"] = (time.perf_counter() - t_api_start) * 1000.0

        rate_limit_note: Optional[str] = None
        if hit_429:
            self.session_max_tries = max(1, effective_tries // 2)
            rate_limit_note = f"Rate limited, using {self.session_max_tries} tries"
            logger.warning(rate_limit_note)

        t_vote = time.perf_counter()
        if not valid_tries:
            err_msg = rate_limit_note or last_err or "Malformed model response"
            res = {"status": "error", "error": err_msg, "found": False}
        else:
            res = vote(
                valid_tries=valid_tries,
                configured_tries=effective_tries,
                early_exited=early_exited,
            )
            if rate_limit_note and res.get("status") == "ok":
                res["agree_text"] = f"{rate_limit_note} · {res.get('agree_text', '')}"
        timings["vote_ms"] = (time.perf_counter() - t_vote) * 1000.0
        return res, timings