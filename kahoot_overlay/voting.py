import json
import re
from typing import Any, Dict, List, Optional, Tuple

VALID_COLORS = ("red", "blue", "yellow", "green")
VALID_TYPES = ("single", "multi", "truefalse", "order")
VALID_POSITIONS = (
    "top-left",
    "top-right",
    "bottom-left",
    "bottom-right",
    "left",
    "right",
    "top",
    "bottom",
    "center",
)

DEFAULT_COLOR_POS = {
    "red": "top-left",
    "blue": "top-right",
    "yellow": "bottom-left",
    "green": "bottom-right",
}


def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def extract_first_json_object(raw_text: str) -> Optional[Dict[str, Any]]:
    """Strip markdown fences and extract the first JSON object from model output."""
    if not raw_text or not isinstance(raw_text, str):
        return None

    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = cleaned.find("{")
    if start == -1:
        return None

    depth = 0
    in_str = False
    escape = False
    for idx in range(start, len(cleaned)):
        ch = cleaned[idx]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if not in_str:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start : idx + 1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            return parsed
                    except Exception:
                        break
    return None


def parse_model_response(raw_text: str, try_index: int = 0, is_temp_zero: bool = False) -> Optional[Dict[str, Any]]:
    """
    Validate and normalize a raw model response string into a structured try dict.
    Returns None if unparseable or invalid.
    """
    obj = extract_first_json_object(raw_text)
    if obj is None:
        return None

    found_raw = obj.get("found", True)
    if isinstance(found_raw, bool):
        found = found_raw
    elif isinstance(found_raw, str):
        found = found_raw.strip().lower() not in ("false", "0", "no")
    else:
        found = bool(found_raw)

    if not found:
        return {
            "found": False,
            "type": "single",
            "answer": (),
            "pos": (),
            "conf": 0,
            "think": str(obj.get("think", ""))[:200],
            "try_index": try_index,
            "is_temp_zero": is_temp_zero,
        }

    qtype = str(obj.get("type", "single")).strip().lower()
    if qtype not in VALID_TYPES:
        qtype = "single"

    ans_raw = obj.get("answer")
    if isinstance(ans_raw, str):
        ans_list = [x.strip().lower() for x in re.split(r"[,+/\s]+", ans_raw) if x.strip()]
    elif isinstance(ans_raw, list):
        ans_list = [str(x).strip().lower() for x in ans_raw if isinstance(x, (str, int, float))]
    else:
        return None

    valid_ans: List[str] = []
    for c in ans_list:
        if c in VALID_COLORS and c not in valid_ans:
            valid_ans.append(c)

    if not valid_ans:
        return None

    pos_raw = obj.get("pos", [])
    if isinstance(pos_raw, str):
        pos_list = [pos_raw.strip().lower()]
    elif isinstance(pos_raw, list):
        pos_list = [str(p).strip().lower() for p in pos_raw]
    else:
        pos_list = []

    color_to_pos: Dict[str, str] = {}
    for idx, c in enumerate(valid_ans):
        if idx < len(pos_list) and pos_list[idx] in VALID_POSITIONS:
            color_to_pos[c] = pos_list[idx]
        else:
            color_to_pos[c] = DEFAULT_COLOR_POS.get(c, "center")

    if qtype == "order":
        choice_tuple = tuple(valid_ans)
    else:
        choice_tuple = tuple(sorted(valid_ans))

    ordered_pos = tuple(color_to_pos[c] for c in choice_tuple)

    conf_raw = obj.get("conf", 50)
    try:
        conf_val = int(round(float(conf_raw)))
    except (ValueError, TypeError):
        conf_val = 50
    conf_val = int(clamp(conf_val, 0, 100))

    return {
        "found": True,
        "type": qtype,
        "answer": choice_tuple,
        "pos": ordered_pos,
        "conf": conf_val,
        "think": str(obj.get("think", ""))[:200],
        "try_index": try_index,
        "is_temp_zero": is_temp_zero,
    }


def parse_stage1_extraction(raw_text: str) -> Optional[Dict[str, Any]]:
    """Parse Stage 1 vision extraction JSON for the two-stage pipeline."""
    obj = extract_first_json_object(raw_text)
    if obj is None:
        return None
    found = obj.get("found", True)
    if isinstance(found, str):
        found = found.strip().lower() not in ("false", "0", "no")
    if not found:
        return {"found": False}
    question = str(obj.get("question", "")).strip()
    options = obj.get("options", {})
    if not isinstance(options, dict):
        options = {}
    positions = obj.get("positions", {})
    if not isinstance(positions, dict):
        positions = {}
    if not question and not options:
        return None
    return {
        "found": True,
        "type": str(obj.get("type", "single")).strip().lower(),
        "question": question,
        "options": {k.lower(): str(v) for k, v in options.items() if k.lower() in VALID_COLORS and v is not None},
        "positions": {k.lower(): str(v).lower() for k, v in positions.items() if k.lower() in VALID_COLORS and v},
    }


def final_confidence(
    winning_confs: List[int],
    winner_votes: int,
    finished_valid_tries: int,
    configured_tries: int,
) -> int:
    """
    Pure confidence calculation per Section 7.4:
    - With tries == 1 (or single valid try when configured_tries == 1):
      final_confidence = clamp(self_conf, 1, 95)
    - Otherwise:
      agreement = winner_votes / finished_valid_tries
      avg_conf = mean(self_conf of the winning tries)
      final_confidence = round(clamp(avg_conf * agreement, 1, 99))
    Never returns 100.
    """
    if not winning_confs or finished_valid_tries <= 0 or winner_votes <= 0:
        return 1

    if configured_tries == 1:
        return int(round(clamp(float(winning_confs[0]), 1.0, 95.0)))

    agreement = float(winner_votes) / float(finished_valid_tries)
    avg_conf = sum(float(c) for c in winning_confs) / float(len(winning_confs))
    val = round(clamp(avg_conf * agreement, 1.0, 99.0))
    return int(clamp(val, 1, 99))


def vote(
    valid_tries: List[Dict[str, Any]],
    configured_tries: int = 3,
    early_exited: bool = False,
) -> Dict[str, Any]:
    """
    Pure voting function over finished valid tries per Section 7.4.
    1. If a majority of finished valid tries report found=False, returns found=False.
    2. Otherwise groups found=True tries by choice tuple.
    3. Winner = choice with most votes. Tie-break: higher sum of self_conf, then temperature=0 try.
    4. Computes final_confidence and detail string.
    """
    if not valid_tries:
        return {
            "status": "error",
            "error": "No valid model responses",
            "found": False,
        }

    total_valid = len(valid_tries)
    not_found_tries = [t for t in valid_tries if not t.get("found", False)]
    found_tries = [t for t in valid_tries if t.get("found", False) and t.get("answer")]

    if len(not_found_tries) > total_valid / 2.0 or not found_tries:
        return {
            "status": "no_question",
            "found": False,
            "winner_votes": len(not_found_tries),
            "finished_valid_tries": total_valid,
            "early_exited": early_exited,
        }

    groups: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}
    for t in found_tries:
        ans = tuple(t["answer"])
        groups.setdefault(ans, []).append(t)

    def sort_key(item: Tuple[Tuple[str, ...], List[Dict[str, Any]]]) -> Tuple[int, int, int]:
        choice, tries_for_choice = item
        vote_count = len(tries_for_choice)
        conf_sum = sum(int(x.get("conf", 0)) for x in tries_for_choice)
        has_temp0 = 1 if any(bool(x.get("is_temp_zero", False)) for x in tries_for_choice) else 0
        return (vote_count, conf_sum, has_temp0)

    winner_choice, winning_tries = max(groups.items(), key=sort_key)
    winner_votes = len(winning_tries)
    finished_valid_tries = len(found_tries)

    winning_confs = [int(x.get("conf", 0)) for x in winning_tries]
    conf_pct = final_confidence(
        winning_confs=winning_confs,
        winner_votes=winner_votes,
        finished_valid_tries=finished_valid_tries,
        configured_tries=configured_tries,
    )

    rep_try = next((x for x in winning_tries if x.get("is_temp_zero")), winning_tries[0])
    pos_tuple = rep_try.get("pos") or tuple(DEFAULT_COLOR_POS.get(c, "center") for c in winner_choice)
    qtype = rep_try.get("type", "single")

    if configured_tries == 1 and finished_valid_tries == 1:
        agree_text = "single try"
    else:
        agree_text = f"{winner_votes} of {finished_valid_tries} tries agree"
        if early_exited:
            agree_text += " (early)"

    return {
        "status": "ok",
        "found": True,
        "type": qtype,
        "answer": list(winner_choice),
        "pos": list(pos_tuple),
        "confidence": conf_pct,
        "winner_votes": winner_votes,
        "finished_valid_tries": finished_valid_tries,
        "early_exited": early_exited,
        "agree_text": agree_text,
    }