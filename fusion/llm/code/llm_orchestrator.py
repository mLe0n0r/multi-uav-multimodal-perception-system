"""
Fuse visual detections and radio transcript into one LLM JSON artefact.

Post-LLM steps (deterministic): validate visual objects, person roles, audio-only
N−V counts, communications (opening check-ins + service basis), counts_by_class.
"""

import argparse
import copy
import json
import re
import sys
import time
import requests
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

_LLM_CODE = Path(__file__).resolve().parent
_FUSION_ROOT = _LLM_CODE.parents[1]
_LLM_DIR = _LLM_CODE.parent
_SLS_DIR = _FUSION_ROOT / "sls"
for path in (_FUSION_ROOT, _LLM_CODE, _SLS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_layout import (
    discover_visual_views,
    is_multi_view_run,
    load_cross_view,
    load_primary_visual,
    load_visual_views,
    llm_output_path,
    transcript_path,
    visual_views_from_cross_view,
)
from communication_demand import (
    RESPONDER_ROLES as DEMAND_RESPONDER_ROLES,
    infer_throughput_need,
)
from fused_counts import (
    VALID_CLASSES,
    _match_list,
    build_fused_object_list,
    deduped_visual_counts,
)

# Re-export for callers that import from llm_orchestrator
__all__ = ("VALID_CLASSES", "deduped_visual_counts")
OLLAMA_MODEL = "gemma4:e2b"
LLM_DIR = _LLM_DIR
DEFAULT_PROMPT = LLM_DIR / "prompts" / "sls_orchestrator_prompt.txt"
MULTIVIEW_PROMPT_ADDON = LLM_DIR / "prompts" / "sls_orchestrator_multiview_addon.txt"
NEAR_FIRE_METERS = 5.0
RESPONDER_ROLES = frozenset({"possible_responder", "firefighter"})
EN_ROUTE_PATTERN = re.compile(
    r"\b(?:en\s+route|on\s+(?:the\s+)?way|responding\s+to|heading\s+(?:to|toward)|"
    r"deployed\s+from|minutes\s+out)\b",
    re.IGNORECASE,
)

THERMAL_SERVICE_PATTERN = re.compile(
    r"thermal\s+imag(?:e|ery)|thermal\s+monitoring|heat\s+spread|heat\s+assessment|\bthermal\b",
    re.IGNORECASE,
)
THERMAL_REQUEST_PATTERN = re.compile(
    r"request\s+thermal|thermal\s+(?:imag(?:e|ery)|data).*(?:assess|spread)|"
    r"(?:need|request).{0,50}thermal",
    re.IGNORECASE,
)
IMAGE_SERVICE_PATTERN = re.compile(
    r"image\s+transfer|visual\s+confirmation|visual\s+assessment|imagery|"
    r"visual\s+confirmation\s+of\s+positions",
    re.IGNORECASE,
)
VIDEO_SERVICE_PATTERN = re.compile(r"\bvideo\b|live\s+video", re.IGNORECASE)
COMMAND_SERVICE_PATTERN = re.compile(
    r"command|coordination|all\s+units|incident\s+assessment",
    re.IGNORECASE,
)
# One opening check-in per speaker; hard cap avoids huge lists on noisy diarization.
MAX_KEY_COMMUNICATIONS_CAP = 6
# Only scan the opening of the transcript for unit check-ins (who speaks / who is addressed).
OPENING_ADDRESSING_WINDOW_SEC = 30.0
RADIO_CHECKIN_ONLY = re.compile(
    r"^(?:command\s+here|engine\s+\d*\s+here|rescue\s+\d*\s+here).{0,40}"
    r"(?:command|engine|listening|finished)\.?$",
    re.IGNORECASE,
)
SERVICE_TO_PATTERN = (
    ("thermal_image", THERMAL_SERVICE_PATTERN),
    ("image_transfer", IMAGE_SERVICE_PATTERN),
    ("video", VIDEO_SERVICE_PATTERN),
    ("command_aggregation", COMMAND_SERVICE_PATTERN),
)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def save_json(data: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def ollama_base_url(ollama_url: str) -> str:
    marker = "/api/"
    if marker in ollama_url:
        return ollama_url.split(marker, 1)[0]
    return ollama_url.rstrip("/")


def ensure_ollama_ready(ollama_url: str) -> None:
    base = ollama_base_url(ollama_url)
    try:
        response = requests.get(f"{base}/api/tags", timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Ollama is not reachable at {base}. "
            "Start the Ollama desktop app (or run `ollama serve`), then retry. "
            "Check models with: ollama list"
        ) from exc


def compact_visual_for_llm(visual_json: Dict[str, Any], view_id: Optional[str] = None) -> Dict[str, Any]:
    """Send only fields the LLM needs; omit large geometry blocks."""
    objects = []
    for obj in visual_json.get("objects", []):
        row = {
            "id": obj.get("id"),
            "class": obj.get("class"),
            "distance_to_fire": obj.get("distance_to_fire"),
            "detection_confidence": obj.get("detection_confidence"),
        }
        if view_id is not None:
            row["view_id"] = view_id
        objects.append(row)
    out = {
        "has_fire": visual_json.get("has_fire"),
        "counts_by_class": visual_json.get("counts_by_class", {}),
        "objects": objects,
    }
    if view_id is not None:
        out["view_id"] = view_id
    return out


def build_fusion_context_from_cross_view(cross_view: Dict[str, Any]) -> Dict[str, Any]:
    """
    Bundle for the LLM from cross_view.json.

    - same_incident true: compact visuals + cross-view matches (fusion).
    - same_incident false: full unaltered visual JSON per view (independent analysis).
    """
    same_incident = bool(cross_view.get("same_incident"))
    view_ids = list(cross_view.get("views", []))
    raw_visuals = cross_view.get("visuals") or {}

    base: Dict[str, Any] = {"views": view_ids}

    if not same_incident:
        # No same_incident key — LLM treats each visual as an independent scene.
        return {
            **base,
            "visuals": {
                view_id: copy.deepcopy(raw_visuals.get(view_id, {}))
                for view_id in view_ids
            },
            "matches": [],
        }

    scene_by_view: Dict[str, Any] = {}
    for view_id in view_ids:
        vis = raw_visuals.get(view_id, {})
        scene_by_view[view_id] = {
            "has_fire": vis.get("has_fire"),
            "counts_by_class": vis.get("counts_by_class", {}),
        }

    views_for_fusion = [
        {**copy.deepcopy(raw_visuals.get(vid, {})), "_view_id": vid} for vid in view_ids
    ]
    fused_objects, _ = build_fused_object_list(views_for_fusion, cross_view)

    return {
        **base,
        "same_incident": True,
        "scene_by_view": scene_by_view,
        "fused_objects": fused_objects,
    }


def visual_entity_keys(
    views: List[Dict[str, Any]],
    matching: Optional[Dict[str, Any]] = None,
) -> Set[Tuple[str, int]]:
    keys: Set[Tuple[str, int]] = set()
    for v in views:
        view_id = v.get("_view_id", "mono")
        for obj in v.get("objects", []):
            vid = resolve_visual_object_id(obj.get("id"), v.get("objects", []))
            if vid is not None:
                keys.add((view_id, vid))
    return keys


def merged_has_fire(views: List[Dict[str, Any]]) -> bool:
    return any(bool(v.get("has_fire")) for v in views)


def compact_transcript_for_llm(transcript_json: Dict[str, Any]) -> Dict[str, Any]:
    """Omit word-level segments to keep the prompt within context limits."""
    segments = []
    for seg in transcript_json.get("segments", []):
        segments.append(
            {
                "speaker": seg.get("speaker"),
                "start": seg.get("start"),
                "end": seg.get("end"),
                "text": (seg.get("text") or "").strip(),
            }
        )
    compact: Dict[str, Any] = {"segments": segments}
    if "analytics" in transcript_json:
        compact["analytics"] = transcript_json["analytics"]
    return compact


def ollama_generate_options() -> Dict[str, Any]:
    return {
        "temperature": 0,
        "num_predict": 4096,
        "repeat_penalty": 1.18,
        "repeat_last_n": 128,
    }


def _ns_to_seconds(ns: Any) -> Optional[float]:
    if ns is None:
        return None
    try:
        return round(float(ns) / 1e9, 2)
    except (TypeError, ValueError):
        return None


def ollama_timing_from_payload(
    payload: Dict[str, Any],
    *,
    wall_clock_sec: float,
    prompt_chars: int,
) -> Dict[str, Any]:
    """Build timing dict from Ollama /api/generate response (durations in ns)."""
    eval_count = int(payload.get("eval_count") or 0)
    prompt_eval_count = int(payload.get("prompt_eval_count") or 0)
    eval_sec = _ns_to_seconds(payload.get("eval_duration"))
    prompt_eval_sec = _ns_to_seconds(payload.get("prompt_eval_duration"))
    load_sec = _ns_to_seconds(payload.get("load_duration"))
    total_sec = _ns_to_seconds(payload.get("total_duration"))

    eval_tps = None
    if eval_sec and eval_sec > 0 and eval_count > 0:
        eval_tps = round(eval_count / eval_sec, 1)

    return {
        "wall_clock_sec": round(wall_clock_sec, 2),
        "ollama_total_sec": total_sec,
        "ollama_load_sec": load_sec,
        "ollama_prompt_eval_sec": prompt_eval_sec,
        "ollama_eval_sec": eval_sec,
        "prompt_tokens": prompt_eval_count,
        "output_tokens": eval_count,
        "output_tokens_per_sec": eval_tps,
        "prompt_chars": prompt_chars,
        "done_reason": payload.get("done_reason"),
    }


def print_ollama_timing(timing: Dict[str, Any], *, attempt: Optional[int] = None) -> None:
    prefix = "Ollama"
    if attempt is not None:
        prefix += f" (attempt {attempt})"
    wall = timing.get("wall_clock_sec")
    prompt_tok = timing.get("prompt_tokens")
    out_tok = timing.get("output_tokens")
    tps = timing.get("output_tokens_per_sec")
    parts = [f"{prefix}: wall {wall}s"]
    if timing.get("ollama_prompt_eval_sec") is not None:
        parts.append(f"prompt {timing['ollama_prompt_eval_sec']}s ({prompt_tok} tok)")
    if timing.get("ollama_eval_sec") is not None:
        line = f"generate {timing['ollama_eval_sec']}s ({out_tok} tok"
        if tps is not None:
            line += f", {tps} tok/s"
        line += ")"
        parts.append(line)
    if timing.get("ollama_load_sec"):
        parts.append(f"model load {timing['ollama_load_sec']}s")
    if timing.get("done_reason"):
        parts.append(f"done={timing['done_reason']}")
    print(", ".join(parts))


def call_ollama(
    transcript_json: Dict[str, Any],
    prompt: str,
    visual_json: Optional[Dict[str, Any]] = None,
    fusion_context: Optional[Dict[str, Any]] = None,
    model: str = OLLAMA_MODEL,
    ollama_url: str = "http://localhost:11434/api/generate",
    strict_json: bool = False,
    *,
    timeout_sec: int = 600,
) -> Tuple[str, Dict[str, Any]]:
    ensure_ollama_ready(ollama_url)

    llm_transcript = compact_transcript_for_llm(transcript_json)
    multi = fusion_context is not None and len(fusion_context.get("views", [])) > 1

    json_rules = (
        "Reply with a single JSON object only. No markdown, no explanation. "
        "Never repeat the same JSON key on consecutive lines. "
    )
    fused_multi = (
        multi and fusion_context is not None and fusion_context.get("same_incident") is True
    )
    if multi:
        if fused_multi:
            json_rules += (
                "Close every array and object; include exactly one objects[] row per "
                "id in FUSION_CONTEXT fused_objects (fused scene entities only). "
                "Do not use view_id or per-camera local ids."
            )
        else:
            json_rules += (
                "Close every array and object; include every object from each view in "
                "FUSION_CONTEXT visuals with its view_id — do not merge across views."
            )
    else:
        json_rules += "Close every array and object; include every visual object id from VISUAL_JSON."
    if strict_json:
        json_rules += " Keep the response compact."

    prompt_text = prompt.strip()
    if multi and MULTIVIEW_PROMPT_ADDON.is_file():
        prompt_text += "\n" + load_text(str(MULTIVIEW_PROMPT_ADDON))

    if multi and fusion_context is not None:
        visual_block = (
            f"FUSION_CONTEXT:\n{json.dumps(fusion_context, ensure_ascii=False)}\n\n"
        )
    else:
        llm_visual = compact_visual_for_llm(visual_json or {})
        visual_block = f"VISUAL_JSON:\n{json.dumps(llm_visual, ensure_ascii=False)}\n\n"

    full_prompt = (
        f"{prompt_text}\n\n"
        f"{json_rules}\n\n"
        f"{visual_block}"
        f"TRANSCRIPT_JSON:\n{json.dumps(llm_transcript, ensure_ascii=False)}"
    )

    prompt_chars = len(full_prompt)
    t0 = time.perf_counter()
    try:
        response = requests.post(
            ollama_url,
            json={
                "model": model,
                "prompt": full_prompt,
                "stream": False,
                "format": "json",
                "options": ollama_generate_options(),
            },
            timeout=timeout_sec,
        )
        response.raise_for_status()
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status == 404:
            raise RuntimeError(
                f"Ollama returned 404 for {ollama_url}. "
                f"Start Ollama and pull the model, e.g. ollama pull {OLLAMA_MODEL}"
            ) from exc
        raise
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Failed to call Ollama at {ollama_url}: {exc}. "
            "Ensure the Ollama app is running (do not start a second `ollama serve` if port 11434 is busy). "
            "Wait for any other LLM run to finish, then retry. Check: ollama list"
        ) from exc

    wall_clock_sec = time.perf_counter() - t0
    payload = response.json()
    timing = ollama_timing_from_payload(
        payload, wall_clock_sec=wall_clock_sec, prompt_chars=prompt_chars
    )
    text = (payload.get("response") or "").strip()
    if not text:
        reason = payload.get("done_reason", "unknown")
        raise RuntimeError(
            "Ollama returned an empty response. "
            f"done_reason={reason}. "
            "The prompt may be too long; this script now sends a compact transcript (no word_segments)."
        )
    return text, timing


def extract_balanced_json_slice(text: str, start: int) -> str:
    """From opening `{` at start, return through the matching `}` or EOF if truncated."""
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return text[start:]


def close_truncated_json_text(json_text: str) -> str:
    """Append closing quotes, brackets, and braces for truncated model JSON."""
    text = json_text.rstrip()
    stack: List[str] = []
    in_string = False
    escape = False
    for char in text:
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            stack.append("}")
        elif char == "[":
            stack.append("]")
        elif char in "}]" and stack and stack[-1] == char:
            stack.pop()
    suffix = ""
    if in_string:
        suffix += '"'
    suffix += "".join(reversed(stack))
    return text + suffix


_INCOMPLETE_KEY_TAIL = re.compile(r',\s*"(?:[^"\\]|\\.)*"\s*:\s*$')
_INCOMPLETE_KEY_NAME = re.compile(r',\s*"[^"]*$')


def strip_incomplete_json_tail(json_text: str) -> str:
    """Drop a trailing partial key so brace closing can yield valid JSON."""
    text = json_text.rstrip()
    changed = True
    while changed and text:
        changed = False
        match = _INCOMPLETE_KEY_TAIL.search(text)
        if match:
            text = text[: match.start()].rstrip()
            changed = True
            continue
        match = _INCOMPLETE_KEY_NAME.search(text)
        if match:
            text = text[: match.start()].rstrip()
            changed = True
            continue
        if text.endswith(":"):
            text = text[:-1].rstrip().rstrip(",").rstrip()
            changed = True
    return text


def repair_truncated_json_text(json_text: str) -> Optional[str]:
    """Best-effort text repair before json.loads."""
    candidates = [json_text]
    closed = close_truncated_json_text(json_text)
    if closed != json_text:
        candidates.append(closed)
    stripped = strip_incomplete_json_tail(json_text)
    if stripped != json_text:
        candidates.extend([stripped, close_truncated_json_text(stripped)])
    seen: Set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            continue
    return None


def extract_json_text(response_text: str) -> str:
    text = response_text.strip()
    if not text:
        raise ValueError("empty response")

    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()

    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found in model output")
    balanced = extract_balanced_json_slice(text, start)
    if balanced.count("{") and balanced.rstrip().endswith("}"):
        return balanced.strip()
    end = text.rfind("}")
    if end > start:
        return text[start : end + 1]
    return text[start:].strip()


def strip_runaway_repetition(json_text: str) -> str:
    """Remove repeated identical key lines from truncated model output."""
    return re.sub(r'(\n\s*"role":\s*"normal",?\s*)+', "\n", json_text)


def extract_balanced_object_blocks(array_text: str) -> List[str]:
    blocks: List[str] = []
    depth = 0
    start: Optional[int] = None
    for index, char in enumerate(array_text):
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start is not None:
                blocks.append(array_text[start : index + 1])
                start = None
    return blocks


def salvage_truncated_json(json_text: str) -> Optional[Dict[str, Any]]:
    """Recover when the model loops or truncates inside objects[]."""
    cleaned = strip_runaway_repetition(json_text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    objects_match = re.search(r'"objects"\s*:\s*\[', cleaned)
    if not objects_match:
        return None

    header = cleaned[: objects_match.end()]
    array_body = cleaned[objects_match.end() :]
    valid_objects: List[Any] = []
    for block in extract_balanced_object_blocks(array_body):
        try:
            valid_objects.append(json.loads(block))
        except json.JSONDecodeError:
            break

    if not valid_objects:
        return None

    head_text = cleaned[: objects_match.start()].rstrip().rstrip(",")
    try:
        head = json.loads(head_text + "}")
    except json.JSONDecodeError:
        return None

    head["objects"] = valid_objects
    return head


def parse_llm_json(response_text: str) -> Dict[str, Any]:
    json_text = extract_json_text(response_text)
    try:
        return json.loads(json_text)
    except json.JSONDecodeError as exc:
        repaired = repair_truncated_json_text(json_text)
        if repaired is not None:
            print("LLM JSON repaired (truncated response; objects filled in post-processing)")
            return json.loads(repaired)
        salvaged = salvage_truncated_json(json_text)
        if salvaged is not None:
            return salvaged
        raise ValueError(
            f"Model returned text that is not valid JSON: {exc}. "
            f"First 200 chars: {json_text[:200]!r}"
        ) from exc


def strip_inference_quotes(text: str) -> str:
    """Remove wrapping quotes — role_inference_basis is inference, not a transcript excerpt."""
    cleaned = " ".join(str(text).split()).strip()
    quote_pairs = (
        ('"', '"'),
        ("'", "'"),
        ("\u201c", "\u201d"),
        ("\u2018", "\u2019"),
    )
    changed = True
    while changed and len(cleaned) >= 2:
        changed = False
        for open_q, close_q in quote_pairs:
            if cleaned.startswith(open_q) and cleaned.endswith(close_q):
                cleaned = cleaned[len(open_q) : -len(close_q)].strip()
                changed = True
                break
    return cleaned


def normalize_role_inference_basis(raw: Any) -> Dict[str, str]:
    """
    Person role justification — always inference (never a quoted transcript excerpt).
    Uses source + text so it is not confused with service_inference_basis quotes.
    """
    if isinstance(raw, dict):
        text = strip_inference_quotes(str(raw.get("text") or ""))
    elif isinstance(raw, str):
        text = strip_inference_quotes(raw)
    elif isinstance(raw, list):
        text = ""
        for item in raw:
            text = strip_inference_quotes(str(item))
            if text:
                break
    else:
        text = ""
    return {"source": "inference", "text": text}


def person_role_fields(
    inferred_role: Any = None,
    role_inference_basis: Any = None,
    role_confidence: Any = None,
) -> Dict[str, Any]:
    return {
        "inferred_role": inferred_role,
        "role_inference_basis": normalize_role_inference_basis(role_inference_basis),
        "role_confidence": role_confidence,
    }


VISUAL_PERCEPTION_FIELDS = (
    "detection_confidence",
    "position",
    "localization_confidence",
    "distance_to_fire",
)


def visual_perception_fields(visual_obj: Dict[str, Any]) -> Dict[str, Any]:
    """Copy perception geometry/confidence from visual.json into llm_output objects."""
    return {
        key: visual_obj[key]
        for key in VISUAL_PERCEPTION_FIELDS
        if key in visual_obj and visual_obj[key] is not None
    }


def semantic_object_from_llm(
    obj_class: str,
    obj: Dict[str, Any],
    *,
    default_role: Optional[str] = None,
    default_basis: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Operational fields for validate_objects; role fields only when class is person."""
    entry: Dict[str, Any] = {
        "risk_level": obj.get("risk_level", "medium"),
    }
    if obj.get("throughput_need"):
        entry["throughput_need"] = obj.get("throughput_need")
    if obj.get("thermal_imagery_consumer"):
        entry["thermal_imagery_consumer"] = True
    if obj_class == "person":
        entry.update(
            person_role_fields(
                obj.get("inferred_role", default_role),
                obj.get("role_inference_basis", default_basis),
                obj.get("role_confidence"),
            )
        )
    return entry


def get_object_by_id(visual_json: Dict[str, Any], object_id: int) -> Optional[Dict[str, Any]]:
    for obj in visual_json.get("objects", []):
        if obj.get("id") == object_id:
            return obj
    return None


_LLM_OBJECT_ID_PATTERN = re.compile(
    r"^(?P<cls>person|normal_vehicle|emergency_vehicle)[_-](?P<idx>\d+)$",
    re.IGNORECASE,
)


def resolve_visual_object_id(
    oid: Any,
    visual_objects: List[Dict[str, Any]],
    *,
    llm_class: Optional[str] = None,
) -> Optional[int]:
    """
    Map an LLM object id to a numeric visual id for this view.

    Accepts ints, numeric strings, and labels like person_1 (1-based index per class).
    """
    if oid is None or isinstance(oid, bool):
        return None

    valid_ids = {
        int(o["id"])
        for o in visual_objects
        if o.get("id") is not None and str(o.get("id")).isdigit()
    }
    if isinstance(oid, int):
        return oid if oid in valid_ids else None

    text = str(oid).strip()
    if text.isdigit():
        vid = int(text)
        return vid if vid in valid_ids else None

    match = _LLM_OBJECT_ID_PATTERN.match(text)
    if not match:
        return None

    cls = (llm_class or match.group("cls")).lower()
    if cls not in VALID_CLASSES:
        return None

    idx = int(match.group("idx"))
    candidates = sorted(
        (o for o in visual_objects if o.get("class") == cls),
        key=lambda o: int(o.get("id", 0)),
    )
    if not candidates:
        return None

    for pick in (idx - 1, idx):
        if 0 <= pick < len(candidates):
            vid = candidates[pick].get("id")
            if vid is not None and int(vid) in valid_ids:
                return int(vid)
    return None


def get_object_by_view_and_id(
    views: List[Dict[str, Any]], view_id: str, object_id: int
) -> Optional[Dict[str, Any]]:
    for v in views:
        if v.get("_view_id", "mono") != view_id:
            continue
        for obj in v.get("objects", []):
            if obj.get("id") == object_id:
                return obj
    return None


def count_objects_by_class(objects: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {cls: 0 for cls in VALID_CLASSES}
    for obj in objects:
        cls = obj.get("class")
        if cls in counts:
            counts[cls] += 1
    return counts


def mentioned_counts_from_transcript(transcript_json: Dict[str, Any]) -> Dict[str, int]:
    """Explicit spoken quantities per class (vehicles); person uses role-aware slots separately."""
    analytics = transcript_json.get("analytics", {}) or {}
    return {
        "normal_vehicle": int(analytics.get("vehicles_mentioned_count", 0) or 0),
        "emergency_vehicle": int(analytics.get("emergency_vehicle_mentioned_count", 0) or 0),
    }


def count_visual_persons_by_role(
    llm_output: Dict[str, Any], visual_json: Dict[str, Any]
) -> tuple[int, int]:
    """Visual persons classified as civilian vs responder after role inference."""
    visual_person_ids = {
        obj.get("id")
        for obj in visual_json.get("objects", [])
        if obj.get("class") == "person" and obj.get("id") is not None
    }
    civilians = 0
    responders = 0
    for obj in llm_output.get("objects", []):
        oid = obj.get("id")
        if oid is None or oid not in visual_person_ids:
            continue
        role = (obj.get("inferred_role") or "").lower()
        if role in RESPONDER_ROLES:
            responders += 1
        else:
            civilians += 1
    return civilians, responders


def person_audio_only_needed(
    llm_output: Dict[str, Any],
    visual_json: Dict[str, Any],
    transcript_json: Dict[str, Any],
) -> int:
    """
    Audio-only persons: gaps per role (civilians / firefighters mentioned vs visually matched).
    Fallback: aggregate people_mentioned_count minus visual person count.
    """
    analytics = transcript_json.get("analytics", {}) or {}
    civilians_n = int(analytics.get("civilians_mentioned_count", 0) or 0)
    firefighters_n = int(analytics.get("firefighters_mentioned_count", 0) or 0)
    responders_n = max(firefighters_n, on_scene_responder_count(transcript_json))
    people_n = int(analytics.get("people_mentioned_count", 0) or 0)

    vis_civ, vis_ff = count_visual_persons_by_role(llm_output, visual_json)
    slots = max(0, civilians_n - vis_civ) + max(0, responders_n - vis_ff)
    if slots > 0:
        return slots

    visual_p = sum(
        1 for obj in visual_json.get("objects", []) if obj.get("class") == "person"
    )
    aggregate_n = people_n if people_n > 0 else (civilians_n + responders_n)
    return max(0, aggregate_n - visual_p)


def parse_distance_meters(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().lower()
    match = re.match(r"([\d.]+)\s*m", text)
    if match:
        return float(match.group(1))
    try:
        return float(text)
    except ValueError:
        return None


def visual_person_distances(visual_json: Dict[str, Any]) -> List[tuple]:
    rows: List[tuple] = []
    for vobj in visual_json.get("objects", []):
        if vobj.get("class") != "person":
            continue
        distance = parse_distance_meters(vobj.get("distance_to_fire"))
        if distance is not None and vobj.get("id") is not None:
            rows.append((vobj["id"], distance))
    return rows


def person_distance_by_id(
    visual_json: Dict[str, Any],
    *,
    fused_objects: Optional[List[Dict[str, Any]]] = None,
) -> Dict[Any, float]:
    if fused_objects:
        distance_by_id: Dict[Any, float] = {}
        for row in fused_objects:
            if row.get("class") != "person":
                continue
            distance = parse_distance_meters(row.get("distance_to_fire"))
            if distance is not None and row.get("id") is not None:
                distance_by_id[row["id"]] = distance
        return distance_by_id
    return dict(visual_person_distances(visual_json))


def normalize_unit_id(token: str) -> str:
    word_map = {"one": "1", "two": "2", "three": "3", "four": "4"}
    cleaned = token.strip().lower()
    return word_map.get(cleaned, cleaned)


def normalize_field_unit(unit_type: str, unit_id: str) -> str:
    unit = unit_type.lower()
    uid = normalize_unit_id(unit_id) if unit_id else ""
    return f"{unit}_{uid}" if uid else unit


def extract_field_speaking_units(text: str) -> Set[str]:
    """
    Field units that speak on the radio (on scene unless en route).

    Uses call-sign phrasing, not diarization labels — e.g. "Command here, engine 1"
    means the speaker is engine 1 at the scene; command is off-scene support.
    """
    cleaned = " ".join(text.split()).strip()
    if not cleaned or EN_ROUTE_PATTERN.search(cleaned):
        return set()

    units: Set[str] = set()
    match = re.match(
        r"^command\s+here,?\s*(?:(engine|rescue)\s+(one|two|\d+)|(alpha|beta))\b",
        cleaned,
        re.IGNORECASE,
    )
    if match:
        if match.group(1):
            units.add(normalize_field_unit(match.group(1), match.group(2)))
        elif match.group(3):
            units.add(normalize_field_unit(match.group(3), ""))

    match = re.match(
        r"^(?:(rescue|engine)\s+(one|two|\d+)|(alpha|beta))\s+here,?\s*(?:engine|rescue|alpha|beta|command)\b",
        cleaned,
        re.IGNORECASE,
    )
    if match:
        if match.group(1):
            units.add(normalize_field_unit(match.group(1), match.group(2)))
        elif match.group(3):
            units.add(normalize_field_unit(match.group(3), ""))
    return units


def on_scene_field_units_from_transcript(transcript_json: Dict[str, Any]) -> Set[str]:
    units: Set[str] = set()
    for segment in transcript_json.get("segments", []):
        text = (segment.get("text") or "").strip()
        if not text or EN_ROUTE_PATTERN.search(text):
            continue
        units |= extract_field_speaking_units(text)
    return units


def on_scene_responder_count(transcript_json: Dict[str, Any]) -> int:
    analytics = transcript_json.get("analytics", {}) or {}
    units = on_scene_field_units_from_transcript(transcript_json)
    if units:
        return len(units)
    firefighters = int(analytics.get("firefighters_mentioned_count", 0) or 0)
    if firefighters > 0:
        return firefighters
    speaker_count = int(analytics.get("speaker_count", 0) or 0)
    if speaker_count >= 2:
        return max(1, speaker_count - 1)
    return 0


def apply_on_scene_person_roles(
    llm_output: Dict[str, Any],
    visual_json: Dict[str, Any],
    transcript_json: Dict[str, Any],
    *,
    fused_objects: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Assign person roles from radio: field units on scene (not command), civilians at safe distance.

    Unless speech says a unit is en route, engine/rescue speakers in radio check-ins are
    treated as on-scene responders. Command is support and never mapped to a visual person.
    """
    analytics = transcript_json.get("analytics", {}) or {}
    responder_count = on_scene_responder_count(transcript_json)
    civilians_count = int(analytics.get("civilians_mentioned_count", 0) or 0)
    units = on_scene_field_units_from_transcript(transcript_json)

    if responder_count <= 0 and not analytics.get("civilians_at_safe_distance_mentioned"):
        return llm_output
    if not llm_output.get("has_fire") and not visual_json.get("has_fire"):
        return llm_output

    distance_by_id = person_distance_by_id(visual_json, fused_objects=fused_objects)
    if not distance_by_id:
        return llm_output

    if not units and responder_count <= 0:
        return apply_proximity_responder_heuristic(
            llm_output,
            visual_json,
            transcript_json,
            fused_objects=fused_objects,
        )

    persons = sorted(distance_by_id.items(), key=lambda row: row[1])
    responder_ids = {oid for oid, _ in persons[:responder_count]}
    remaining = [(oid, dist) for oid, dist in persons if oid not in responder_ids]
    remaining.sort(key=lambda row: -row[1])
    civilian_ids = {oid for oid, _ in remaining[:civilians_count]}

    unit_note = ", ".join(sorted(units)) if units else "field radio unit"

    for obj in llm_output.get("objects", []):
        oid = obj.get("id")
        if oid is None or obj.get("class") != "person" or oid not in distance_by_id:
            continue
        if oid in responder_ids:
            obj["inferred_role"] = "firefighter"
            obj["role_inference_basis"] = {
                "source": "inference",
                "text": (
                    f"{unit_note} on scene per radio (command is off-scene support); "
                    "assigned to closest person near fire"
                ),
            }
            obj["role_confidence"] = 0.85
        elif oid in civilian_ids:
            obj["inferred_role"] = "civilian"
            obj["role_inference_basis"] = {
                "source": "inference",
                "text": (
                    f"radio reports {civilians_count} civilian(s) at safe distance; "
                    "matched to persons farther from fire"
                ),
            }
            obj["role_confidence"] = 0.75

    return llm_output


def apply_proximity_responder_heuristic(
    llm_output: Dict[str, Any],
    visual_json: Dict[str, Any],
    transcript_json: Dict[str, Any],
    *,
    fused_objects: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """When civilians are at safe distance, persons near fire -> responder; farther -> civilian."""
    analytics = transcript_json.get("analytics", {}) or {}
    if not analytics.get("civilians_at_safe_distance_mentioned"):
        return llm_output
    if not llm_output.get("has_fire") and not visual_json.get("has_fire"):
        return llm_output

    if fused_objects:
        distance_by_id = person_distance_by_id(visual_json, fused_objects=fused_objects)
    else:
        distance_by_id = dict(visual_person_distances(visual_json))
    if not distance_by_id:
        return llm_output

    for obj in llm_output.get("objects", []):
        oid = obj.get("id")
        if oid is None or obj.get("class") != "person" or oid not in distance_by_id:
            continue
        distance = distance_by_id[oid]
        if distance <= NEAR_FIRE_METERS:
            obj["inferred_role"] = "possible_responder"
            note = "near fire while radio reports civilians at safe distance"
        else:
            obj["inferred_role"] = "civilian"
            note = "farther from fire while radio reports civilians at safe distance"
        obj["role_inference_basis"] = {"source": "inference", "text": note}
        if obj.get("role_confidence") is None:
            obj["role_confidence"] = 0.7
    return llm_output


def transcript_requests_thermal_imagery(transcript_json: Dict[str, Any]) -> bool:
    """True when speech explicitly requests thermal imagery (not only scene-level mention)."""
    for seg in transcript_json.get("segments", []):
        text = (seg.get("text") or seg.get("message") or "").strip()
        if not text:
            continue
        if THERMAL_REQUEST_PATTERN.search(text):
            return True
        if THERMAL_SERVICE_PATTERN.search(text) and re.search(
            r"\brequest\b|\bneed\b|\bsend\b|\bprovide\b",
            text,
            re.IGNORECASE,
        ):
            return True
    return False


def pick_thermal_imagery_consumer_index(objects: List[Dict[str, Any]]) -> Optional[int]:
    """Person most likely to receive thermal feed (responder at fire, else closest)."""
    responders: List[tuple] = []
    by_distance: List[tuple] = []
    for i, obj in enumerate(objects):
        if obj.get("audio_only") or obj.get("class") != "person":
            continue
        role = str(obj.get("inferred_role") or "").lower()
        dist = parse_distance_meters(obj.get("distance_to_fire"))
        if role in DEMAND_RESPONDER_ROLES:
            responders.append((dist if dist is not None else 0.0, i))
        elif dist is not None:
            by_distance.append((dist, i))
    if responders:
        responders.sort(key=lambda x: x[0])
        return responders[0][1]
    if by_distance:
        by_distance.sort(key=lambda x: x[0])
        return by_distance[0][1]
    return None


def apply_throughput_need_heuristics(
    llm_output: Dict[str, Any],
    transcript_json: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Role + inferred services determine throughput_need (not distance).

    Distance is used only for risk_level (SLS) and for civilian-at-risk / vehicle
    operational proximity at the communication policy layer.
    """
    thermal_requested = transcript_requests_thermal_imagery(transcript_json)
    comms = llm_output.get("communications", {}) or {}
    scene_services = list(comms.get("service_types") or ["voice"])
    if not thermal_requested and "thermal_image" in scene_services:
        for item in comms.get("service_inference_basis") or []:
            text = (item.get("text") or "") if isinstance(item, dict) else ""
            if THERMAL_REQUEST_PATTERN.search(text) or (
                THERMAL_SERVICE_PATTERN.search(text)
                and re.search(r"\brequest\b", text, re.IGNORECASE)
            ):
                thermal_requested = True
                break

    consumer_idx = (
        pick_thermal_imagery_consumer_index(llm_output.get("objects", []))
        if thermal_requested
        else None
    )
    has_fire = bool(llm_output.get("has_fire"))

    for i, obj in enumerate(llm_output.get("objects", [])):
        if obj.get("audio_only"):
            obj["throughput_need"] = "low"
            obj.pop("thermal_imagery_consumer", None)
            continue
        obj.pop("thermal_imagery_consumer", None)
        cls = obj.get("class", "")
        is_consumer = bool(thermal_requested and i == consumer_idx and cls == "person")
        if is_consumer:
            obj["thermal_imagery_consumer"] = True
            basis = obj.get("role_inference_basis")
            note = "radio requests thermal imagery for this unit (engine/responder)"
            if isinstance(basis, dict):
                prior = (basis.get("text") or "").strip()
                basis = {**basis, "text": f"{prior}; {note}" if prior else note}
            else:
                basis = {"source": "transcript", "text": note}
            obj["role_inference_basis"] = basis
        obj["throughput_need"] = infer_throughput_need(
            obj,
            scene_services,
            thermal_consumer=is_consumer,
            has_fire=has_fire,
        )
    return llm_output


def default_audio_only_person_role(transcript_json: Dict[str, Any]) -> str:
    analytics = transcript_json.get("analytics", {}) or {}
    civilians = int(analytics.get("civilians_mentioned_count", 0) or 0)
    firefighters = int(analytics.get("firefighters_mentioned_count", 0) or 0)
    responders = max(firefighters, on_scene_responder_count(transcript_json))
    if civilians > 0 and civilians >= firefighters:
        return "civilian"
    if responders > 0:
        return "firefighter"
    return "unknown_person"


def audio_only_reason(
    cls: str,
    mentioned_n: int,
    visual_n: int,
    *,
    civilians_mentioned: Optional[int] = None,
    visual_civilian: Optional[int] = None,
) -> str:
    if cls == "person" and civilians_mentioned is not None and visual_civilian is not None:
        return (
            f"radio reports {civilians_mentioned} civilian(s) at safe distance; "
            f"{visual_civilian} matched visually as civilian"
        )
    label = cls.replace("_", " ")
    return (
        f"radio explicitly mentions {mentioned_n} {label}(s); "
        f"{visual_n} detected visually"
    )


def make_audio_only_entry(
    cls: str,
    transcript_json: Dict[str, Any],
    mentioned_n: int,
    visual_n: int,
    llm_obj: Optional[Dict[str, Any]] = None,
    *,
    civilians_mentioned: Optional[int] = None,
    visual_civilian: Optional[int] = None,
    role_override: Optional[str] = None,
    reason_override: Optional[str] = None,
) -> Dict[str, Any]:
    base = dict(llm_obj or {})
    base["reason"] = reason_override or (
        audio_only_reason(
            cls,
            mentioned_n,
            visual_n,
            civilians_mentioned=civilians_mentioned,
            visual_civilian=visual_civilian,
        )
    )
    base.setdefault("risk_level", "medium")
    base.setdefault("throughput_need", "medium")
    if cls == "person":
        base["inferred_role"] = role_override or base.get(
            "inferred_role", default_audio_only_person_role(transcript_json)
        )
    entry: Dict[str, Any] = {
        "id": None,
        "class": cls,
        "audio_only": True,
        "reason": base["reason"],
        **semantic_object_from_llm(
            cls,
            base,
            default_role=base.get("inferred_role") if cls == "person" else None,
        ),
    }
    return entry


def reconcile_audio_only_by_class(
    llm_output: Dict[str, Any],
    visual_json: Dict[str, Any],
    transcript_json: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Vehicles/emergency: N_radio − V_visual per class.
    Persons: role-aware gaps (e.g. 2 civilians on radio, 1 seen as civilian → +1 audio_only).
    """
    mentioned = mentioned_counts_from_transcript(transcript_json)
    visual_counts = count_objects_by_class(visual_json.get("objects", []))
    analytics = transcript_json.get("analytics", {}) or {}

    objects = llm_output.get("objects", [])
    non_audio = [obj for obj in objects if not obj.get("audio_only")]
    audio_by_class: Dict[str, List[Dict[str, Any]]] = {cls: [] for cls in VALID_CLASSES}
    for obj in objects:
        cls = obj.get("class")
        if obj.get("audio_only") and cls in audio_by_class:
            audio_by_class[cls].append(obj)

    merged_audio: List[Dict[str, Any]] = []
    for cls in VALID_CLASSES:
        if cls == "person":
            civ_n = int(analytics.get("civilians_mentioned_count", 0) or 0)
            ff_n = int(analytics.get("firefighters_mentioned_count", 0) or 0)
            responder_n = max(ff_n, on_scene_responder_count(transcript_json))
            visual_n = visual_counts.get("person", 0)
            vis_civ, _ = count_visual_persons_by_role(llm_output, visual_json)
            _, vis_ff = count_visual_persons_by_role(llm_output, visual_json)
            civ_needed = max(0, civ_n - vis_civ)
            ff_needed = max(0, responder_n - vis_ff)
            needed = civ_needed + ff_needed
            mentioned_n = int(analytics.get("people_mentioned_count", 0) or 0) or (
                civ_n + responder_n
            )
            person_kw = {"civilians_mentioned": civ_n, "visual_civilian": vis_civ}
            role_targets = (["civilian"] * civ_needed) + (["firefighter"] * ff_needed)
        else:
            person_kw = {}
            needed = max(0, mentioned.get(cls, 0) - visual_counts.get(cls, 0))
            mentioned_n = mentioned.get(cls, 0)
            visual_n = visual_counts.get(cls, 0)
            role_targets = []
        kept: List[Dict[str, Any]] = []
        for obj in audio_by_class[cls]:
            if len(kept) >= needed:
                break
            role_override = None
            reason_override = None
            if cls == "person":
                idx = len(kept)
                role_override = role_targets[idx] if idx < len(role_targets) else None
                if role_override == "firefighter":
                    reason_override = (
                        f"radio indicates {responder_n} on-scene responder(s); "
                        f"{vis_ff} matched visually as firefighter"
                    )
            kept.append(
                make_audio_only_entry(
                    cls,
                    transcript_json,
                    mentioned_n,
                    visual_n,
                    obj,
                    role_override=role_override,
                    reason_override=reason_override,
                    **person_kw,
                )
            )
        while len(kept) < needed:
            role_override = None
            reason_override = None
            if cls == "person":
                idx = len(kept)
                role_override = role_targets[idx] if idx < len(role_targets) else None
                if role_override == "firefighter":
                    reason_override = (
                        f"radio indicates {responder_n} on-scene responder(s); "
                        f"{vis_ff} matched visually as firefighter"
                    )
            kept.append(
                make_audio_only_entry(
                    cls,
                    transcript_json,
                    mentioned_n,
                    visual_n,
                    role_override=role_override,
                    reason_override=reason_override,
                    **person_kw,
                )
            )
        merged_audio.extend(kept)

    llm_output["objects"] = non_audio + merged_audio
    llm_output["objects"].sort(
        key=lambda o: (o.get("id") is None, o.get("id") if o.get("id") is not None else 0)
    )
    return llm_output


def max_audio_only_by_class(
    visual_json: Dict[str, Any],
    transcript_json: Optional[Dict[str, Any]] = None,
    llm_output: Optional[Dict[str, Any]] = None,
    views: Optional[List[Dict[str, Any]]] = None,
    matching: Optional[Dict[str, Any]] = None,
) -> Dict[str, int]:
    """How many audio_only extras are allowed per class."""
    view_list = views if views else [visual_json]
    if len(view_list) > 1:
        visual_counts = deduped_visual_counts(view_list, matching)
    else:
        visual_counts = count_objects_by_class(visual_json.get("objects", []))
    mentioned = mentioned_counts_from_transcript(transcript_json or {})
    slots = {
        cls: max(0, mentioned.get(cls, 0) - visual_counts.get(cls, 0))
        for cls in VALID_CLASSES
    }
    if llm_output is not None and transcript_json is not None:
        slots["person"] = person_audio_only_needed(
            llm_output, visual_json, transcript_json
        )
    else:
        analytics = (transcript_json or {}).get("analytics", {}) or {}
        people_n = int(analytics.get("people_mentioned_count", 0) or 0)
        civ_n = int(analytics.get("civilians_mentioned_count", 0) or 0)
        ff_n = int(analytics.get("firefighters_mentioned_count", 0) or 0)
        aggregate = people_n if people_n > 0 else (civ_n + ff_n)
        slots["person"] = max(0, aggregate - visual_counts.get("person", 0))
    return slots


def normalize_legacy_output(llm_output: Dict[str, Any]) -> Dict[str, Any]:
    """Map older entity-based LLM output to visual-style objects[]."""
    if llm_output.get("objects"):
        return llm_output

    incident = llm_output.get("incident_summary", {}) or {}
    if incident:
        llm_output.setdefault("has_fire", incident.get("has_fire"))
        llm_output.setdefault("summary", incident.get("summary"))
        llm_output.setdefault("communications", incident.get("audio_context", {}))

        object_counts = incident.get("object_counts", {})
        if object_counts and "counts_by_class" not in llm_output:
            counts = {}
            for cls in VALID_CLASSES:
                block = object_counts.get(cls, {})
                if isinstance(block, dict):
                    counts[cls] = int(block.get("estimated_scene_total", block.get("observed_visual", 0)) or 0)
                else:
                    counts[cls] = int(block or 0)
            llm_output["counts_by_class"] = counts

    objects: List[Dict[str, Any]] = []
    for entity in llm_output.get("entities", []):
        oid = entity.get("source_object_id", entity.get("id"))
        objects.append(
            {
                "id": oid,
                "class": entity.get("input_class", entity.get("class")),
                "inferred_role": entity.get("inferred_role"),
                "role_inference_basis": entity.get("role_inference_basis", []),
                "role_confidence": entity.get("role_confidence"),
                "risk_level": entity.get("risk_level"),
                "throughput_need": entity.get("throughput_need"),
            }
        )

    for item in llm_output.get("additional_entities_from_audio", []):
        count = int(item.get("count", 1) or 1)
        for _ in range(max(1, count)):
            objects.append(
                {
                    "id": None,
                    "class": item.get("input_class", item.get("class")),
                    "audio_only": True,
                    "reason": item.get("reason", ""),
                    "inferred_role": item.get("inferred_role"),
                    "risk_level": item.get("risk_level", "medium"),
                    "throughput_need": item.get("throughput_need", "medium"),
                }
            )

    if objects:
        llm_output["objects"] = objects
    return llm_output


def _validate_fused_objects(
    llm_output: Dict[str, Any],
    view_list: List[Dict[str, Any]],
    matching: Dict[str, Any],
    transcript_json: Optional[Dict[str, Any]],
    visual_json: Dict[str, Any],
) -> Dict[str, Any]:
    fused_objects, key_map = build_fused_object_list(view_list, matching)
    n_fused = len(fused_objects)
    valid_ids = set(range(n_fused))
    semantic_by_fid: Dict[int, Dict[str, Any]] = {}
    audio_only_slots = max_audio_only_by_class(
        visual_json, transcript_json, llm_output, views=view_list, matching=matching
    )

    for obj in llm_output.get("objects", []):
        if obj.get("audio_only"):
            continue
        fid = obj.get("id")
        if fid is not None and obj.get("view_id") is not None:
            mapped = key_map.get((str(obj["view_id"]), int(fid)))
            if mapped is not None:
                fid = mapped
        if fid is None:
            continue
        fid = int(fid)
        if fid not in valid_ids:
            continue
        semantic_by_fid[fid] = obj

    cleaned: List[Dict[str, Any]] = []
    for row in fused_objects:
        fid = int(row["id"])
        cls = row.get("class")
        sem = semantic_by_fid.get(fid, {})
        cleaned.append(
            {
                "id": fid,
                "class": cls,
                **visual_perception_fields(row),
                **semantic_object_from_llm(cls, sem),
            }
        )

    for obj in llm_output.get("objects", []):
        if obj.get("audio_only"):
            cls = obj.get("class")
            if cls not in VALID_CLASSES:
                continue
            if audio_only_slots.get(cls, 0) <= 0:
                continue
            audio_only_slots[cls] -= 1
            default_role = "civilian" if cls == "person" else None
            cleaned.append(
                {
                    "id": None,
                    "class": cls,
                    "audio_only": True,
                    "reason": obj.get("reason", ""),
                    **semantic_object_from_llm(cls, obj, default_role=default_role),
                }
            )

    cleaned.sort(key=lambda o: (o.get("id") is None, o.get("id") if o.get("id") is not None else 0))
    llm_output["objects"] = cleaned
    return llm_output


def validate_objects(
    llm_output: Dict[str, Any],
    visual_json: Dict[str, Any],
    transcript_json: Optional[Dict[str, Any]] = None,
    views: Optional[List[Dict[str, Any]]] = None,
    matching: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    view_list = views if views else [visual_json]
    multi = len(view_list) > 1
    if multi and matching and matching.get("same_incident"):
        return _validate_fused_objects(
            llm_output, view_list, matching, transcript_json, visual_json
        )
    valid_keys = visual_entity_keys(view_list, matching) if multi else None
    valid_ids = {obj.get("id") for obj in visual_json.get("objects", [])}
    cleaned: List[Dict[str, Any]] = []
    seen_visual_ids: set = set()
    seen_entity_keys: Set[Tuple[str, int]] = set()
    audio_only_slots = max_audio_only_by_class(visual_json, transcript_json, llm_output, views=view_list, matching=matching)

    for obj in llm_output.get("objects", []):
        raw_oid = obj.get("id")
        if raw_oid is not None:
            if multi:
                view_id = obj.get("view_id") or view_list[0].get("_view_id", "mono")
                view_objs = next(
                    (v.get("objects", []) for v in view_list if v.get("_view_id", "mono") == view_id),
                    [],
                )
                oid = resolve_visual_object_id(
                    raw_oid, view_objs, llm_class=obj.get("class")
                )
                if oid is None:
                    continue
                key = (str(view_id), oid)
                if key not in valid_keys or key in seen_entity_keys:
                    continue
                original = get_object_by_view_and_id(view_list, str(view_id), oid)
                if original is None:
                    continue
                seen_entity_keys.add(key)
                cls = original.get("class", obj.get("class"))
                entry = {
                    "id": oid,
                    "view_id": view_id,
                    "class": cls,
                    **visual_perception_fields(original),
                    **semantic_object_from_llm(cls, obj),
                }
                if obj.get("also_seen_in"):
                    entry["also_seen_in"] = obj.get("also_seen_in")
                cleaned.append(entry)
            else:
                oid = resolve_visual_object_id(
                    raw_oid, visual_json.get("objects", []), llm_class=obj.get("class")
                )
                if oid is None or oid not in valid_ids or oid in seen_visual_ids:
                    continue
                original = get_object_by_id(visual_json, oid)
                if original is None:
                    continue
                seen_visual_ids.add(oid)
                cls = original.get("class", obj.get("class"))
                cleaned.append(
                    {
                        "id": oid,
                        "class": cls,
                        **visual_perception_fields(original),
                        **semantic_object_from_llm(cls, obj),
                    }
                )
        elif obj.get("audio_only"):
            cls = obj.get("class")
            if cls not in VALID_CLASSES:
                continue
            if audio_only_slots.get(cls, 0) <= 0:
                continue
            audio_only_slots[cls] -= 1
            default_role = "civilian" if cls == "person" else None
            cleaned.append(
                {
                    "id": None,
                    "class": cls,
                    "audio_only": True,
                    "reason": obj.get("reason", ""),
                    **semantic_object_from_llm(cls, obj, default_role=default_role),
                }
            )

    for v in view_list:
        view_id = v.get("_view_id", "mono")
        for vobj in v.get("objects", []):
            oid = vobj.get("id")
            if oid is None:
                continue
            if multi:
                vid = resolve_visual_object_id(oid, v.get("objects", []))
                if vid is None:
                    continue
                key = (str(view_id), vid)
                if key in seen_entity_keys:
                    continue
                seen_entity_keys.add(key)
                oid = vid
            elif oid in seen_visual_ids:
                continue
            else:
                seen_visual_ids.add(oid)
            cls = vobj.get("class")
            cleaned.append(
                {
                    "id": oid,
                    **({"view_id": view_id} if multi else {}),
                    "class": cls,
                    **visual_perception_fields(vobj),
                    **semantic_object_from_llm(
                        cls,
                        {"risk_level": "medium", "throughput_need": "low"},
                        default_role="unknown_person" if cls == "person" else None,
                        default_basis=(
                            {"source": "inference", "text": "visual_detection"}
                            if cls == "person"
                            else None
                        ),
                    ),
                }
            )

    cleaned.sort(
        key=lambda o: (
            o.get("id") is None,
            o.get("view_id", ""),
            o.get("id") if o.get("id") is not None else 0,
        )
    )
    llm_output["objects"] = cleaned
    return llm_output


def normalize_all_person_fields(llm_output: Dict[str, Any]) -> Dict[str, Any]:
    for obj in llm_output.get("objects", []):
        if obj.get("class") == "person":
            obj["role_inference_basis"] = normalize_role_inference_basis(
                obj.get("role_inference_basis")
            )
    return llm_output


def transcript_basis_entry(text: str) -> Dict[str, str]:
    """Verbatim radio line — source marks it as transcript, no wrapping quotes in text."""
    cleaned = " ".join(str(text).split()).strip()
    return {"source": "transcript", "text": cleaned}


def normalize_service_inference_basis(raw: Any) -> List[Dict[str, str]]:
    """Normalize LLM output: transcript vs inference, never with quote characters in text."""
    if not isinstance(raw, list):
        raw = [raw] if raw else []

    out: List[Dict[str, str]] = []
    seen: set = set()
    for item in raw:
        if isinstance(item, dict):
            source = item.get("source", "inference")
            if source not in ("transcript", "inference"):
                source = "inference"
            text = " ".join(str(item.get("text") or "").split()).strip()
            if source == "inference":
                text = strip_inference_quotes(text)
        elif isinstance(item, str):
            text = " ".join(item.split()).strip()
            source = "transcript" if text.startswith('"') and text.endswith('"') else "inference"
            text = strip_inference_quotes(text)
        else:
            continue
        if not text:
            continue
        key = (source, text)
        if key in seen:
            continue
        seen.add(key)
        out.append({"source": source, "text": text})
    return out


def find_segment_matching(
    transcript_json: Dict[str, Any], pattern: re.Pattern[str]
) -> Optional[Dict[str, Any]]:
    for segment in transcript_json.get("segments", []):
        text = (segment.get("text") or "").strip()
        if text and pattern.search(text):
            return segment
    return None


def extract_radio_addressing(text: str) -> Optional[str]:
    """Opening check-in phrase from a segment (may be embedded in a longer utterance)."""
    cleaned = " ".join(text.split()).strip()
    if not cleaned:
        return None
    if cleaned.lower().rstrip(".") in ("listening", "finished"):
        return None
    if RADIO_CHECKIN_ONLY.match(cleaned):
        return cleaned
    prefix_patterns = (
        r"^(Engine\s+(?:one|\d+)\s+here,?\s*command)\.",
        r"^(Engine\s+one\s+here,?\s*command)\.",
        r"^(Command\s+here,?\s*engine\s+(?:one|\d+))\.",
        r"^(Command\s+here,?\s*rescue\s+(?:two|\d+))\.",
        r"^(Rescue\s+(?:two|\d+)\s+here,?\s*command)\.",
        r"^(Rescue\s+(?:two|\d+)\s+here,?\s*engine\s+(?:one|\d+))\.",
        r"^(Engine\s+(?:one|\d+)\s+here,?\s*rescue\s+(?:two|\d+))\.",
    )
    for pattern in prefix_patterns:
        match = re.match(pattern, cleaned, re.IGNORECASE)
        if match:
            return match.group(1) + "."
    return None


def is_radio_addressing_excerpt(text: str) -> bool:
    return extract_radio_addressing(text) is not None


def format_addressing_excerpt(text: str) -> str:
    cleaned = " ".join(text.split()).strip()
    if cleaned and cleaned[-1] not in ".!?":
        cleaned += "."
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def segment_to_key_communication(segment: Dict[str, Any]) -> Dict[str, Any]:
    raw = (segment.get("text") or "").strip()
    addressing = extract_radio_addressing(raw) or raw
    entry: Dict[str, Any] = {"text": format_addressing_excerpt(addressing)}
    if segment.get("speaker") is not None:
        entry["speaker"] = segment["speaker"]
    return entry


def dedupe_key_communications_by_speaker(
    items: List[Dict[str, Any]], *, limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Keep at most one key_communications entry per speaker (chronological order)."""
    deduped: List[Dict[str, Any]] = []
    seen_speakers: set = set()
    for item in items:
        speaker = item.get("speaker")
        if speaker is not None:
            if speaker in seen_speakers:
                continue
            seen_speakers.add(speaker)
        deduped.append(item)
        if limit is not None and len(deduped) >= limit:
            break
    return deduped


def normalize_key_communication_item(item: Any) -> Optional[Dict[str, Any]]:
    if isinstance(item, dict):
        raw_text = (item.get("text") or "").strip()
        speaker = item.get("speaker")
    elif isinstance(item, str):
        raw_text = item.strip()
        speaker = None
    else:
        return None

    if not is_radio_addressing_excerpt(raw_text):
        return None

    return segment_to_key_communication({"text": raw_text, "speaker": speaker})


def distinct_speaker_count(transcript_json: Dict[str, Any]) -> int:
    speakers = {
        segment.get("speaker")
        for segment in transcript_json.get("segments", [])
        if segment.get("speaker")
    }
    return len(speakers)


def key_communications_limit(
    speaker_count: int, candidates: Optional[List[Dict[str, Any]]] = None
) -> int:
    """One opening check-in per detected speaker, capped for safety."""
    if speaker_count > 0:
        return min(speaker_count, MAX_KEY_COMMUNICATIONS_CAP)
    if candidates:
        distinct = len(
            {segment.get("speaker") for segment in candidates if segment.get("speaker")}
        )
        if distinct > 0:
            return min(distinct, MAX_KEY_COMMUNICATIONS_CAP)
    return min(2, MAX_KEY_COMMUNICATIONS_CAP)


def _opening_window_segments(
    segments: List[Dict[str, Any]], anchor_start: float
) -> List[Dict[str, Any]]:
    windowed: List[Dict[str, Any]] = []
    for segment in segments:
        start = float(segment.get("start") or 0.0)
        if start - anchor_start > OPENING_ADDRESSING_WINDOW_SEC:
            break
        windowed.append(segment)
    return windowed


def _append_distinct_speaker_segments(
    chosen: List[Dict[str, Any]],
    used_speakers: set,
    segments: List[Dict[str, Any]],
    *,
    limit: int,
    require_addressing: bool,
) -> None:
    for segment in segments:
        if len(chosen) >= limit:
            break
        speaker = segment.get("speaker")
        if not speaker or speaker in used_speakers:
            continue
        raw_text = (segment.get("text") or "").strip()
        if not raw_text:
            continue
        if require_addressing and not is_radio_addressing_excerpt(raw_text):
            continue
        chosen.append(segment)
        used_speakers.add(speaker)


def pick_opening_addressing_communications(
    segments: List[Dict[str, Any]], speaker_count: int
) -> List[Dict[str, Any]]:
    """First unit check-ins in chronological order; exactly one excerpt per speaker."""
    if not segments:
        return []

    anchor_start = float(segments[0].get("start") or 0.0)
    opening_segments = _opening_window_segments(segments, anchor_start)
    candidates = [
        segment
        for segment in opening_segments
        if is_radio_addressing_excerpt((segment.get("text") or "").strip())
    ]

    limit = key_communications_limit(speaker_count, segments)
    chosen: List[Dict[str, Any]] = []
    used_speakers: set = set()

    _append_distinct_speaker_segments(
        chosen, used_speakers, candidates, limit=limit, require_addressing=False
    )
    if len(chosen) < limit:
        _append_distinct_speaker_segments(
            chosen,
            used_speakers,
            opening_segments,
            limit=limit,
            require_addressing=False,
        )
    if len(chosen) < limit:
        _append_distinct_speaker_segments(
            chosen, used_speakers, segments, limit=limit, require_addressing=False
        )

    if not chosen:
        return []

    return [
        segment_to_key_communication(seg)
        for seg in chosen[:limit]
    ]


def reconcile_key_communications(
    comms: Dict[str, Any], transcript_json: Dict[str, Any]
) -> None:
    """Opening radio check-ins (one per speaker, up to speaker_count), no time."""
    analytics = transcript_json.get("analytics", {}) or {}
    speaker_count = int(
        comms.get("speaker_count")
        or analytics.get("speaker_count")
        or distinct_speaker_count(transcript_json)
        or 0
    )

    segments = transcript_json.get("segments", []) or []
    picked = pick_opening_addressing_communications(segments, speaker_count)
    if picked:
        comms["key_communications"] = picked
        return

    limit = key_communications_limit(speaker_count)
    fallback: List[Dict[str, Any]] = []
    raw = comms.get("key_communications", [])
    if isinstance(raw, list):
        for item in raw:
            entry = normalize_key_communication_item(item)
            if entry:
                fallback.append(entry)
    comms["key_communications"] = dedupe_key_communications_by_speaker(
        fallback, limit=limit
    )


def service_type_for_basis_text(text: str) -> Optional[str]:
    """Map a basis excerpt to the scene service it supports."""
    if THERMAL_REQUEST_PATTERN.search(text) or THERMAL_SERVICE_PATTERN.search(text):
        return "thermal_image"
    if IMAGE_SERVICE_PATTERN.search(text):
        return "image_transfer"
    if VIDEO_SERVICE_PATTERN.search(text):
        return "video"
    if COMMAND_SERVICE_PATTERN.search(text):
        return "command_aggregation"
    return None


def transcript_contains_excerpt(transcript_json: Dict[str, Any], excerpt: str) -> bool:
    needle = " ".join(str(excerpt).split()).strip().lower()
    if not needle:
        return False
    for segment in transcript_json.get("segments", []):
        hay = " ".join(str(segment.get("text") or "").split()).strip().lower()
        if needle in hay or hay in needle:
            return True
    return needle in transcript_text(transcript_json).lower()


def filter_service_inference_basis(
    basis: List[Dict[str, str]],
    service_set: Set[str],
    transcript_json: Dict[str, Any],
) -> List[Dict[str, str]]:
    """Keep only basis entries for active non-voice services with transcript support."""
    non_voice = {svc for svc in service_set if svc != "voice"}
    if not non_voice:
        return []

    kept: List[Dict[str, str]] = []
    seen: set = set()
    for item in normalize_service_inference_basis(basis):
        service = service_type_for_basis_text(item.get("text", ""))
        if service not in non_voice:
            continue
        if item.get("source") == "transcript" and not transcript_contains_excerpt(
            transcript_json, item.get("text", "")
        ):
            continue
        key = (item.get("source"), item.get("text"))
        if key in seen:
            continue
        seen.add(key)
        kept.append(item)
    return kept


def reconcile_service_inference_basis(
    comms: Dict[str, Any], transcript_json: Dict[str, Any]
) -> None:
    """One basis entry per non-voice service: {source: transcript, text: verbatim line}."""
    types = comms.get("service_types", []) or []
    if not isinstance(types, list):
        types = [types]
    service_set = set(types)
    non_voice = {svc for svc in service_set if svc != "voice"}
    if not non_voice:
        comms["service_inference_basis"] = []
        return

    basis: List[Dict[str, str]] = []
    seen: set = set()
    for service, pattern in SERVICE_TO_PATTERN:
        if service not in service_set or service == "voice":
            continue
        segment = find_segment_matching(transcript_json, pattern)
        if segment:
            entry = transcript_basis_entry(segment.get("text", ""))
            key = (entry["source"], entry["text"])
            if key not in seen:
                seen.add(key)
                basis.append(entry)

    if not basis:
        basis = filter_service_inference_basis(
            comms.get("service_inference_basis", []),
            service_set,
            transcript_json,
        )
    else:
        basis = normalize_service_inference_basis(basis)
    comms["service_inference_basis"] = basis


def transcript_text(transcript_json: Dict[str, Any]) -> str:
    parts: List[str] = []
    for segment in transcript_json.get("segments", []):
        text = segment.get("text", "")
        if text:
            parts.append(str(text))
    return " ".join(parts)


def enrich_scene_service_types(
    comms: Dict[str, Any], transcript_json: Dict[str, Any]
) -> None:
    """Backstop for prompt rules: ensure voice + mandatory thermal/image/video from speech."""
    text = transcript_text(transcript_json)
    types = comms.get("service_types", comms.get("service_type"))
    if types is None:
        types = []
    if not isinstance(types, list):
        types = [types]
    merged = set(types)
    merged.add("voice")
    if THERMAL_SERVICE_PATTERN.search(text):
        merged.add("thermal_image")
    if IMAGE_SERVICE_PATTERN.search(text):
        merged.add("image_transfer")
    if VIDEO_SERVICE_PATTERN.search(text):
        merged.add("video")
    comms["service_types"] = sorted(merged)


def reconcile_communications(
    llm_output: Dict[str, Any],
    transcript_json: Dict[str, Any],
) -> Dict[str, Any]:
    analytics = transcript_json.get("analytics", {}) or {}
    comms = llm_output.setdefault("communications", {})

    if "speaker_count" not in comms and "speaker_count" in analytics:
        comms["speaker_count"] = analytics["speaker_count"]

    for key in (
        "people_mentioned_count",
        "vehicles_mentioned_count",
        "emergency_vehicle_mentioned_count",
    ):
        comms.pop(key, None)

    enrich_scene_service_types(comms, transcript_json)
    types = comms.get("service_types", ["voice"])
    if not isinstance(types, list):
        types = [types]
    comms["service_types"] = types or ["voice"]
    comms.pop("service_type", None)

    reconcile_key_communications(comms, transcript_json)
    reconcile_service_inference_basis(comms, transcript_json)
    return llm_output


# Fields that belong only under communications, not at llm_output root.
_COMMUNICATIONS_ROOT_KEYS = frozenset(
    {
        "speaker_count",
        "key_communications",
        "service_types",
        "service_type",
        "service_inference_basis",
        "people_mentioned_count",
        "vehicles_mentioned_count",
        "emergency_vehicle_mentioned_count",
    }
)


def consolidate_communications_root(llm_output: Dict[str, Any]) -> Dict[str, Any]:
    """Move communications fields from JSON root into communications (then remove duplicates)."""
    comms = llm_output.setdefault("communications", {})
    for key in _COMMUNICATIONS_ROOT_KEYS:
        if key in llm_output and key not in comms:
            comms[key] = llm_output[key]
        llm_output.pop(key, None)
    return llm_output


def strip_object_ids_from_output(llm_output: Dict[str, Any]) -> Dict[str, Any]:
    """Final llm_output listing: objects are identified by class/position, not numeric id."""
    llm_output.pop("scenario_priority", None)
    for obj in llm_output.get("objects", []):
        obj.pop("id", None)
        obj.pop("view_id", None)
    return llm_output


def reconcile_counts_by_class(
    llm_output: Dict[str, Any],
    visual_json: Dict[str, Any],
    views: Optional[List[Dict[str, Any]]] = None,
    matching: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    view_list = views if views else [visual_json]
    if len(view_list) > 1:
        visual_counts = deduped_visual_counts(view_list, matching)
    else:
        visual_counts = visual_json.get("counts_by_class", {}) or {}
    audio_only_by_class = {cls: 0 for cls in VALID_CLASSES}

    for obj in llm_output.get("objects", []):
        if obj.get("id") is None and obj.get("audio_only"):
            cls = obj.get("class")
            if cls in audio_only_by_class:
                audio_only_by_class[cls] += 1

    counts = {}
    for cls in VALID_CLASSES:
        observed = int(visual_counts.get(cls, 0) or 0)
        counts[cls] = observed + audio_only_by_class[cls]

    llm_output["counts_by_class"] = counts
    if len(view_list) > 1:
        llm_output.setdefault("has_fire", merged_has_fire(view_list))
    else:
        llm_output.setdefault("has_fire", visual_json.get("has_fire"))
    return llm_output


def add_metadata(
    llm_output: Dict[str, Any],
    transcript_json_path: str,
    model: str,
    visual_json_path: Optional[str] = None,
    run_dir: Optional[str] = None,
    multi_view: bool = False,
    ollama_timing: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "transcript_json_path": transcript_json_path,
        "llm_model": model,
        "schema": "visual_json_extended",
        "traffic_demand_policy": "not_applied_in_llm_orchestrator",
        "multi_view": multi_view,
    }
    if visual_json_path:
        meta["visual_json_path"] = visual_json_path
    if run_dir:
        meta["run_dir"] = run_dir
    if ollama_timing:
        meta["ollama_timing"] = ollama_timing
    llm_output["_metadata"] = meta
    return llm_output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fuse visual and audio into visual-style JSON via Ollama (Gemma)."
    )
    parser.add_argument(
        "--run-dir",
        help="Run folder (output/<scenario>/<run_id>); discovers perception/ and fusion/",
    )
    parser.add_argument("--visual-json", help="Single visual.json (mono run; overrides run-dir)")
    parser.add_argument("--transcript-json", help="transcript.json path")
    parser.add_argument(
        "--prompt",
        default=str(DEFAULT_PROMPT),
        help="Orchestrator prompt file",
    )
    parser.add_argument("--output", help="llm_output.json path (default: <run-dir>/fusion/llm_output.json)")
    parser.add_argument(
        "--model",
        default=OLLAMA_MODEL,
        help=f"Ollama model tag (default: {OLLAMA_MODEL})",
    )
    parser.add_argument("--ollama-url", default="http://localhost:11434/api/generate")

    args = parser.parse_args()

    if not args.run_dir and not args.visual_json:
        parser.error("Provide --run-dir or --visual-json")

    run_dir: Optional[Path] = Path(args.run_dir) if args.run_dir else None
    multi_view = False
    cross_view: Optional[Dict[str, Any]] = None
    fusion_context: Optional[Dict[str, Any]] = None

    if args.visual_json:
        visual_json = load_json(args.visual_json)
        visual_json.setdefault("_view_id", "mono")
        views = [visual_json]
        transcript_path_str = args.transcript_json
        if not transcript_path_str and run_dir:
            transcript_path_str = str(transcript_path(run_dir))
        if not transcript_path_str:
            parser.error("Provide --transcript-json or --run-dir with perception/transcript.json")
        transcript_json = load_json(transcript_path_str)
        output_path = Path(args.output) if args.output else None
        if output_path is None and run_dir:
            output_path = llm_output_path(run_dir)
        if output_path is None:
            parser.error("Provide --output or --run-dir")
    else:
        assert run_dir is not None
        views = load_visual_views(run_dir)
        if not views:
            raise FileNotFoundError(f"No visual JSON under {run_dir}/perception/")
        visual_json = views[0]
        for v in views:
            v.setdefault("_view_id", v.get("_view_id", "mono"))
        transcript_json = load_json(str(transcript_path(run_dir)))
        output_path = Path(args.output) if args.output else llm_output_path(run_dir)
        multi_view = is_multi_view_run(run_dir)
        if multi_view:
            cross_view = load_cross_view(run_dir)
            if cross_view is None:
                raise FileNotFoundError(
                    f"Multi-view run requires {run_dir}/fusion/cross_view.json — "
                    "run fusion/matching/cross_view_match.py first"
                )
            views = visual_views_from_cross_view(cross_view)
            visual_json = views[0]
            fusion_context = build_fusion_context_from_cross_view(cross_view)
            if cross_view.get("same_incident"):
                n_match = len(_match_list(cross_view))
                print(
                    f"Multi-view fusion (same_incident): LLM reads FUSION_CONTEXT "
                    f"({n_match} cross-view matches + transcript)"
                )
            else:
                print("Multi-view independent analysis (different incidents)")

    model_name = args.model
    response_text = ""
    llm_output: Optional[Dict[str, Any]] = None
    last_error: Optional[Exception] = None
    ollama_timing: Optional[Dict[str, Any]] = None

    prompt = load_text(str(Path(args.prompt)))
    max_attempts = 3
    timeout_sec = 600
    print(f"Calling Ollama ({model_name}) at {args.ollama_url} ...")

    for attempt in range(1, max_attempts + 1):
        try:
            response_text, attempt_timing = call_ollama(
                transcript_json=transcript_json,
                prompt=prompt,
                visual_json=visual_json if not multi_view else None,
                fusion_context=fusion_context,
                model=model_name,
                ollama_url=args.ollama_url,
                strict_json=attempt > 1,
                timeout_sec=timeout_sec,
            )
            print_ollama_timing(
                attempt_timing, attempt=attempt if max_attempts > 1 else None
            )
            ollama_timing = attempt_timing
            llm_output = parse_llm_json(response_text)
            if attempt > 1:
                print(f"LLM JSON parsed on attempt {attempt}/{max_attempts}")
            break
        except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < max_attempts:
                print(f"LLM attempt {attempt}/{max_attempts} failed ({exc}); retrying...")
                continue

    if llm_output is None:
        if response_text:
            debug_path = output_path.with_name(output_path.stem + "_raw_llm.txt")
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            debug_path.write_text(response_text, encoding="utf-8")
            print(f"Raw model output saved to {debug_path}")
        assert last_error is not None
        raise last_error
    llm_output = normalize_legacy_output(llm_output)
    llm_output = validate_objects(
        llm_output,
        visual_json,
        transcript_json,
        views=views if multi_view else None,
        matching=cross_view,
    )
    fused_for_proximity: Optional[List[Dict[str, Any]]] = None
    if multi_view and cross_view and cross_view.get("same_incident"):
        fused_for_proximity, _ = build_fused_object_list(views, cross_view)
    llm_output = apply_on_scene_person_roles(
        llm_output,
        visual_json,
        transcript_json,
        fused_objects=fused_for_proximity,
    )
    llm_output = apply_throughput_need_heuristics(llm_output, transcript_json)
    llm_output = reconcile_audio_only_by_class(llm_output, visual_json, transcript_json)
    llm_output = reconcile_communications(llm_output, transcript_json)
    llm_output = reconcile_counts_by_class(
        llm_output, visual_json, views=views if multi_view else None, matching=cross_view
    )
    llm_output = normalize_all_person_fields(llm_output)
    llm_output = consolidate_communications_root(llm_output)
    llm_output = strip_object_ids_from_output(llm_output)

    save_json(llm_output, str(output_path))
    timing_note = ""
    if ollama_timing and ollama_timing.get("wall_clock_sec") is not None:
        timing_note = f", wall={ollama_timing['wall_clock_sec']}s"
    print(
        f"LLM output saved to {output_path} "
        f"(model={model_name}, multi_view={multi_view}{timing_note})"
    )


if __name__ == "__main__":
    main()
