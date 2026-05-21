"""
Fuse visual detections and radio transcript into one LLM JSON artefact.

Post-LLM steps (deterministic): validate visual objects, person roles, audio-only
N−V counts, communications (opening check-ins + service basis), counts_by_class.
"""

import argparse
import json
import re
import requests
from pathlib import Path
from typing import Any, Dict, List, Optional


VALID_CLASSES = ("person", "normal_vehicle", "emergency_vehicle")
OLLAMA_MODEL = "gemma4:e4b"
FUSION_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPT = FUSION_DIR / "prompts" / "sls_orchestrator_prompt.txt"
NEAR_FIRE_METERS = 5.0
RESPONDER_ROLES = frozenset({"possible_responder", "firefighter"})

THERMAL_SERVICE_PATTERN = re.compile(
    r"thermal\s+imag(?:e|ery)|thermal\s+monitoring|heat\s+spread|heat\s+assessment|\bthermal\b",
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


def compact_visual_for_llm(visual_json: Dict[str, Any]) -> Dict[str, Any]:
    """Send only fields the LLM needs; omit large geometry blocks."""
    objects = []
    for obj in visual_json.get("objects", []):
        objects.append(
            {
                "id": obj.get("id"),
                "class": obj.get("class"),
                "distance_to_fire": obj.get("distance_to_fire"),
                "detection_confidence": obj.get("detection_confidence"),
            }
        )
    return {
        "has_fire": visual_json.get("has_fire"),
        "counts_by_class": visual_json.get("counts_by_class", {}),
        "objects": objects,
    }


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
        "num_predict": 8192,
        "repeat_penalty": 1.18,
        "repeat_last_n": 128,
    }


def call_ollama(
    visual_json: Dict[str, Any],
    transcript_json: Dict[str, Any],
    prompt: str,
    model: str = OLLAMA_MODEL,
    ollama_url: str = "http://localhost:11434/api/generate",
    strict_json: bool = False,
) -> str:
    ensure_ollama_ready(ollama_url)

    llm_visual = compact_visual_for_llm(visual_json)
    llm_transcript = compact_transcript_for_llm(transcript_json)

    json_rules = (
        "Reply with a single JSON object only. No markdown, no explanation. "
        "Never repeat the same JSON key on consecutive lines. "
        "Close every array and object; include every visual object id from VISUAL_JSON."
    )
    if strict_json:
        json_rules += " Keep the response compact."

    full_prompt = (
        f"{prompt.strip()}\n\n"
        f"{json_rules}\n\n"
        f"VISUAL_JSON:\n{json.dumps(llm_visual, ensure_ascii=False)}\n\n"
        f"TRANSCRIPT_JSON:\n{json.dumps(llm_transcript, ensure_ascii=False)}"
    )

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
            timeout=600,
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
            f"Failed to call Ollama at {ollama_url}. Start Ollama and confirm the model exists: ollama list"
        ) from exc

    payload = response.json()
    text = (payload.get("response") or "").strip()
    if not text:
        reason = payload.get("done_reason", "unknown")
        raise RuntimeError(
            "Ollama returned an empty response. "
            f"done_reason={reason}. "
            "The prompt may be too long; this script now sends a compact transcript (no word_segments)."
        )
    return text


def extract_json_text(response_text: str) -> str:
    text = response_text.strip()
    if not text:
        raise ValueError("empty response")

    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found in model output")
    return text[start : end + 1]


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
        "throughput_need": obj.get("throughput_need", "medium"),
    }
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
    people_n = int(analytics.get("people_mentioned_count", 0) or 0)

    vis_civ, vis_ff = count_visual_persons_by_role(llm_output, visual_json)
    slots = max(0, civilians_n - vis_civ) + max(0, firefighters_n - vis_ff)
    if slots > 0:
        return slots

    visual_p = sum(
        1 for obj in visual_json.get("objects", []) if obj.get("class") == "person"
    )
    aggregate_n = people_n if people_n > 0 else (civilians_n + firefighters_n)
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


def apply_proximity_responder_heuristic(
    llm_output: Dict[str, Any],
    visual_json: Dict[str, Any],
    transcript_json: Dict[str, Any],
) -> Dict[str, Any]:
    """When civilians are at safe distance, persons near fire -> responder; farther -> civilian."""
    analytics = transcript_json.get("analytics", {}) or {}
    if not analytics.get("civilians_at_safe_distance_mentioned"):
        return llm_output
    if not visual_json.get("has_fire"):
        return llm_output

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


def default_audio_only_person_role(transcript_json: Dict[str, Any]) -> str:
    analytics = transcript_json.get("analytics", {}) or {}
    civilians = int(analytics.get("civilians_mentioned_count", 0) or 0)
    firefighters = int(analytics.get("firefighters_mentioned_count", 0) or 0)
    if civilians > 0 and civilians >= firefighters:
        return "civilian"
    if firefighters > 0:
        return "possible_responder"
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
) -> Dict[str, Any]:
    base = dict(llm_obj or {})
    base.setdefault(
        "reason",
        audio_only_reason(
            cls,
            mentioned_n,
            visual_n,
            civilians_mentioned=civilians_mentioned,
            visual_civilian=visual_civilian,
        ),
    )
    base.setdefault("risk_level", "medium")
    base.setdefault("throughput_need", "medium")
    if cls == "person":
        base.setdefault(
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
            needed = person_audio_only_needed(llm_output, visual_json, transcript_json)
            mentioned_n = int(analytics.get("people_mentioned_count", 0) or 0) or (
                int(analytics.get("civilians_mentioned_count", 0) or 0)
                + int(analytics.get("firefighters_mentioned_count", 0) or 0)
            )
            visual_n = visual_counts.get("person", 0)
            vis_civ, _ = count_visual_persons_by_role(llm_output, visual_json)
            civ_n = int(analytics.get("civilians_mentioned_count", 0) or 0)
            person_kw = {"civilians_mentioned": civ_n, "visual_civilian": vis_civ}
        else:
            person_kw = {}
            needed = max(0, mentioned.get(cls, 0) - visual_counts.get(cls, 0))
            mentioned_n = mentioned.get(cls, 0)
            visual_n = visual_counts.get(cls, 0)
        kept: List[Dict[str, Any]] = []
        for obj in audio_by_class[cls]:
            if len(kept) >= needed:
                break
            kept.append(
                make_audio_only_entry(
                    cls,
                    transcript_json,
                    mentioned_n,
                    visual_n,
                    obj,
                    **person_kw,
                )
            )
        while len(kept) < needed:
            kept.append(
                make_audio_only_entry(
                    cls,
                    transcript_json,
                    mentioned_n,
                    visual_n,
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
) -> Dict[str, int]:
    """How many audio_only extras are allowed per class."""
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
        llm_output.setdefault("scenario_priority", incident.get("scenario_priority"))
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


def validate_objects(
    llm_output: Dict[str, Any],
    visual_json: Dict[str, Any],
    transcript_json: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    valid_ids = {obj.get("id") for obj in visual_json.get("objects", [])}
    cleaned: List[Dict[str, Any]] = []
    seen_visual_ids: set = set()
    audio_only_slots = max_audio_only_by_class(visual_json, transcript_json, llm_output)

    for obj in llm_output.get("objects", []):
        oid = obj.get("id")
        if oid is not None:
            if oid not in valid_ids or oid in seen_visual_ids:
                continue
            original = get_object_by_id(visual_json, oid)
            if original is None:
                continue
            seen_visual_ids.add(oid)
            cls = original.get("class", obj.get("class"))
            entry = {
                "id": oid,
                "class": cls,
                **semantic_object_from_llm(cls, obj),
            }
            cleaned.append(entry)
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

    for vobj in visual_json.get("objects", []):
        oid = vobj.get("id")
        if oid is None or oid in seen_visual_ids:
            continue
        cls = vobj.get("class")
        cleaned.append(
            {
                "id": oid,
                "class": cls,
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

    cleaned.sort(key=lambda o: (o.get("id") is None, o.get("id") if o.get("id") is not None else 0))
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


def is_radio_addressing_excerpt(text: str) -> bool:
    """Short opening lines that identify caller and addressee (e.g. 'Engine 1 here, command.')."""
    cleaned = " ".join(text.split()).strip()
    if not cleaned:
        return False
    if cleaned.lower().rstrip(".") in ("listening", "finished"):
        return False
    if RADIO_CHECKIN_ONLY.match(cleaned):
        return True
    if re.match(r"^command\s+here,?\s*engine\s+(?:one|\d+)\.?$", cleaned, re.IGNORECASE):
        return True
    if re.match(r"^rescue\s+(?:one|\d+)\s+here,?\s*command\.?$", cleaned, re.IGNORECASE):
        return True
    return False


def format_addressing_excerpt(text: str) -> str:
    cleaned = " ".join(text.split()).strip()
    if cleaned and cleaned[-1] not in ".!?":
        cleaned += "."
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def segment_to_key_communication(segment: Dict[str, Any]) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "text": format_addressing_excerpt((segment.get("text") or "").strip())
    }
    if segment.get("speaker") is not None:
        entry["speaker"] = segment["speaker"]
    return entry


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


def pick_opening_addressing_communications(
    segments: List[Dict[str, Any]], speaker_count: int
) -> List[Dict[str, Any]]:
    """First unit check-ins in chronological order; one per speaker when possible."""
    if not segments:
        return []

    anchor_start = float(segments[0].get("start") or 0.0)
    candidates: List[Dict[str, Any]] = []
    for segment in segments:
        start = float(segment.get("start") or 0.0)
        if start - anchor_start > OPENING_ADDRESSING_WINDOW_SEC:
            break
        raw_text = (segment.get("text") or "").strip()
        if is_radio_addressing_excerpt(raw_text):
            candidates.append(segment)

    limit = key_communications_limit(speaker_count, candidates)
    chosen: List[Dict[str, Any]] = []
    used_speakers: set = set()

    for segment in candidates:
        speaker = segment.get("speaker")
        if not speaker or speaker in used_speakers:
            continue
        chosen.append(segment)
        used_speakers.add(speaker)
        if len(chosen) >= limit:
            break

    # Fill remaining slots if fewer speakers than limit (duplicate-speaker check-ins).
    for segment in candidates:
        if len(chosen) >= limit:
            break
        if segment in chosen:
            continue
        chosen.append(segment)

    return [segment_to_key_communication(seg) for seg in chosen]


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
            if len(fallback) >= limit:
                break
    comms["key_communications"] = fallback


def reconcile_service_inference_basis(
    comms: Dict[str, Any], transcript_json: Dict[str, Any]
) -> None:
    """One basis entry per non-voice service: {source: transcript, text: verbatim line}."""
    types = comms.get("service_types", []) or []
    if not isinstance(types, list):
        types = [types]
    service_set = set(types)

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
        basis = normalize_service_inference_basis(comms.get("service_inference_basis", []))
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


def reconcile_counts_by_class(
    llm_output: Dict[str, Any],
    visual_json: Dict[str, Any],
) -> Dict[str, Any]:
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
    llm_output.setdefault("has_fire", visual_json.get("has_fire"))
    return llm_output


def add_metadata(
    llm_output: Dict[str, Any],
    visual_json_path: str,
    transcript_json_path: str,
    model: str,
) -> Dict[str, Any]:
    llm_output["_metadata"] = {
        "visual_json_path": visual_json_path,
        "transcript_json_path": transcript_json_path,
        "llm_model": model,
        "schema": "visual_json_extended",
        "traffic_demand_policy": "not_applied_in_llm_orchestrator",
    }
    return llm_output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fuse visual and audio into visual-style JSON via Ollama (Gemma)."
    )
    parser.add_argument("--visual-json", required=True)
    parser.add_argument("--transcript-json", required=True)
    parser.add_argument(
        "--prompt",
        default=str(DEFAULT_PROMPT),
        help="Orchestrator prompt file",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--model",
        default=OLLAMA_MODEL,
        help=f"Ollama model tag (default: {OLLAMA_MODEL})",
    )
    parser.add_argument("--ollama-url", default="http://localhost:11434/api/generate")

    args = parser.parse_args()

    prompt = load_text(str(Path(args.prompt)))
    visual_json = load_json(args.visual_json)
    transcript_json = load_json(args.transcript_json)

    response_text = ""
    llm_output: Optional[Dict[str, Any]] = None
    last_error: Optional[Exception] = None
    max_attempts = 3

    for attempt in range(1, max_attempts + 1):
        try:
            response_text = call_ollama(
                visual_json=visual_json,
                transcript_json=transcript_json,
                prompt=prompt,
                model=args.model,
                ollama_url=args.ollama_url,
                strict_json=attempt > 1,
            )
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
            debug_path = Path(args.output).with_name(
                Path(args.output).stem + "_raw_llm.txt"
            )
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            debug_path.write_text(response_text, encoding="utf-8")
            print(f"Raw model output saved to {debug_path}")
        assert last_error is not None
        raise last_error
    llm_output = normalize_legacy_output(llm_output)
    llm_output = validate_objects(llm_output, visual_json, transcript_json)
    llm_output = apply_proximity_responder_heuristic(
        llm_output, visual_json, transcript_json
    )
    llm_output = reconcile_audio_only_by_class(llm_output, visual_json, transcript_json)
    llm_output = reconcile_communications(llm_output, transcript_json)
    llm_output = reconcile_counts_by_class(llm_output, visual_json)
    llm_output = normalize_all_person_fields(llm_output)
    llm_output = add_metadata(
        llm_output=llm_output,
        visual_json_path=args.visual_json,
        transcript_json_path=args.transcript_json,
        model=args.model,
    )

    save_json(llm_output, args.output)
    print(
        f"LLM output saved to {args.output} "
        f"(model={args.model}, prompt={Path(args.prompt).name})"
    )


if __name__ == "__main__":
    main()
