import argparse
import json
import re
import requests
from pathlib import Path
from typing import Any, Dict, List, Optional


VALID_CLASSES = ("person", "normal_vehicle", "emergency_vehicle")
CIVILIAN_ROLES = frozenset({"civilian", "unknown_person", "person_near_incident"})
RESPONDER_ROLES = frozenset({"possible_responder", "firefighter"})
NEAR_FIRE_METERS = 5.0
MIN_PERSON_DISTANCE_SPREAD = 3.0


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


def call_qwen(
    visual_json: Dict[str, Any],
    transcript_json: Dict[str, Any],
    prompt: str,
    model: str = "qwen2.5:latest",
    ollama_url: str = "http://localhost:11434/api/generate",
) -> str:
    ensure_ollama_ready(ollama_url)

    llm_visual = compact_visual_for_llm(visual_json)
    llm_transcript = compact_transcript_for_llm(transcript_json)

    full_prompt = (
        f"{prompt.strip()}\n\n"
        "Reply with a single JSON object only. No markdown, no explanation.\n\n"
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
                "options": {"temperature": 0, "num_predict": 4096},
            },
            timeout=600,
        )
        response.raise_for_status()
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status == 404:
            raise RuntimeError(
                f"Ollama returned 404 for {ollama_url}. "
                "Start Ollama and use an installed model, e.g. --model qwen2.5:latest"
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


def parse_llm_json(response_text: str) -> Dict[str, Any]:
    json_text = extract_json_text(response_text)
    try:
        return json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Model returned text that is not valid JSON: {exc}. "
            f"First 200 chars: {json_text[:200]!r}"
        ) from exc


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
    analytics = transcript_json.get("analytics", {}) or {}
    return {
        "person": int(analytics.get("people_mentioned_count", 0) or 0),
        "normal_vehicle": int(analytics.get("vehicles_mentioned_count", 0) or 0),
        "emergency_vehicle": int(analytics.get("emergency_vehicle_mentioned_count", 0) or 0),
    }


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
    """Person closest to fire may be a responder when radio says civilians are at safe distance."""
    analytics = transcript_json.get("analytics", {}) or {}
    if not analytics.get("civilians_at_safe_distance_mentioned"):
        return llm_output
    if not visual_json.get("has_fire"):
        return llm_output

    person_dists = visual_person_distances(visual_json)
    if len(person_dists) < 2:
        return llm_output

    person_dists.sort(key=lambda row: row[1])
    closest_id, closest_distance = person_dists[0]
    farthest_distance = person_dists[-1][1]
    if closest_distance > NEAR_FIRE_METERS:
        return llm_output
    if (farthest_distance - closest_distance) < MIN_PERSON_DISTANCE_SPREAD:
        return llm_output

    for obj in llm_output.get("objects", []):
        if obj.get("id") != closest_id or obj.get("class") != "person":
            continue
        obj["inferred_role"] = "possible_responder"
        basis = obj.get("role_inference_basis", [])
        if not isinstance(basis, list):
            basis = []
        note = (
            "closest person to fire while radio reports civilians at safe distance"
        )
        if note not in basis:
            basis.append(note)
        obj["role_inference_basis"] = basis
        if obj.get("role_confidence") is None:
            obj["role_confidence"] = 0.7
        break
    return llm_output


def count_visual_civilian_matches(
    llm_output: Dict[str, Any],
    visual_json: Dict[str, Any],
) -> int:
    visual_person_ids = {
        obj.get("id")
        for obj in visual_json.get("objects", [])
        if obj.get("class") == "person" and obj.get("id") is not None
    }
    matched = 0
    for obj in llm_output.get("objects", []):
        oid = obj.get("id")
        if oid is None or oid not in visual_person_ids:
            continue
        role = (obj.get("inferred_role") or "").lower()
        if role in RESPONDER_ROLES:
            continue
        if role in CIVILIAN_ROLES or not role:
            matched += 1
    return matched


def reconcile_person_fusion(
    llm_output: Dict[str, Any],
    visual_json: Dict[str, Any],
    transcript_json: Dict[str, Any],
) -> Dict[str, Any]:
    """Align person roles and audio_only civilians with explicit radio counts."""
    llm_output = apply_proximity_responder_heuristic(
        llm_output, visual_json, transcript_json
    )

    analytics = transcript_json.get("analytics", {}) or {}
    civilians_mentioned = int(analytics.get("civilians_mentioned_count", 0) or 0)

    objects = llm_output.get("objects", [])
    without_person_audio = [
        obj
        for obj in objects
        if not (obj.get("audio_only") and obj.get("class") == "person")
    ]

    if civilians_mentioned <= 0:
        llm_output["objects"] = without_person_audio
        return llm_output

    visual_civilian_count = count_visual_civilian_matches(llm_output, visual_json)
    needed_audio_civilian = max(0, civilians_mentioned - visual_civilian_count)

    existing_person_audio = [
        obj
        for obj in objects
        if obj.get("audio_only") and obj.get("class") == "person"
    ]
    kept_person_audio: List[Dict[str, Any]] = []
    for obj in existing_person_audio:
        if len(kept_person_audio) >= needed_audio_civilian:
            break
        obj.setdefault("inferred_role", "civilian")
        kept_person_audio.append(obj)

    while len(kept_person_audio) < needed_audio_civilian:
        kept_person_audio.append(
            {
                "id": None,
                "class": "person",
                "audio_only": True,
                "inferred_role": "civilian",
                "reason": (
                    f"radio reports {civilians_mentioned} civilian(s) at safe distance; "
                    f"only {visual_civilian_count} matched visually after role inference"
                ),
                "risk_level": "medium",
                "throughput_need": "medium",
            }
        )

    merged = without_person_audio + kept_person_audio
    merged.sort(
        key=lambda o: (o.get("id") is None, o.get("id") if o.get("id") is not None else 0)
    )
    llm_output["objects"] = merged
    return llm_output


def max_audio_only_by_class(
    visual_json: Dict[str, Any],
    transcript_json: Optional[Dict[str, Any]] = None,
) -> Dict[str, int]:
    """How many audio_only extras are allowed per class (explicit mention count minus visual)."""
    visual_counts = count_objects_by_class(visual_json.get("objects", []))
    mentioned = mentioned_counts_from_transcript(transcript_json or {})
    slots: Dict[str, int] = {}
    for cls in VALID_CLASSES:
        if cls == "person":
            slots[cls] = 0
            continue
        slots[cls] = max(0, mentioned.get(cls, 0) - visual_counts.get(cls, 0))
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
    audio_only_slots = max_audio_only_by_class(visual_json, transcript_json)

    for obj in llm_output.get("objects", []):
        oid = obj.get("id")
        if oid is not None:
            if oid not in valid_ids or oid in seen_visual_ids:
                continue
            original = get_object_by_id(visual_json, oid)
            if original is None:
                continue
            seen_visual_ids.add(oid)
            cleaned.append(
                {
                    "id": oid,
                    "class": original.get("class", obj.get("class")),
                    "inferred_role": obj.get("inferred_role"),
                    "role_inference_basis": obj.get("role_inference_basis", [])
                    if isinstance(obj.get("role_inference_basis"), list)
                    else [],
                    "role_confidence": obj.get("role_confidence"),
                    "risk_level": obj.get("risk_level", "medium"),
                    "throughput_need": obj.get("throughput_need", "medium"),
                }
            )
        elif obj.get("audio_only"):
            cls = obj.get("class")
            if cls not in VALID_CLASSES:
                continue
            if cls == "person":
                cleaned.append(
                    {
                        "id": None,
                        "class": "person",
                        "audio_only": True,
                        "reason": obj.get("reason", ""),
                        "inferred_role": obj.get("inferred_role", "civilian"),
                        "risk_level": obj.get("risk_level", "medium"),
                        "throughput_need": obj.get("throughput_need", "medium"),
                    }
                )
                continue
            if audio_only_slots.get(cls, 0) <= 0:
                continue
            audio_only_slots[cls] -= 1
            cleaned.append(
                {
                    "id": None,
                    "class": cls,
                    "audio_only": True,
                    "reason": obj.get("reason", ""),
                    "inferred_role": obj.get("inferred_role"),
                    "risk_level": obj.get("risk_level", "medium"),
                    "throughput_need": obj.get("throughput_need", "medium"),
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
                "inferred_role": {
                    "person": "unknown_person",
                    "normal_vehicle": "background_vehicle",
                    "emergency_vehicle": "emergency_vehicle",
                }.get(cls, "unknown_person"),
                "role_inference_basis": ["visual_detection"],
                "role_confidence": None,
                "risk_level": "medium",
                "throughput_need": "low",
            }
        )

    cleaned.sort(key=lambda o: (o.get("id") is None, o.get("id") if o.get("id") is not None else 0))
    llm_output["objects"] = cleaned
    return llm_output


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

    comms.setdefault("key_communications", [])

    types = comms.get("service_types", comms.get("service_type"))
    if types is None:
        types = ["voice"]
    if not isinstance(types, list):
        types = [types]
    comms["service_types"] = types or ["voice"]
    comms.pop("service_type", None)
    comms.setdefault("service_inference_basis", [])
    if not isinstance(comms["service_inference_basis"], list):
        comms["service_inference_basis"] = [str(comms["service_inference_basis"])]

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
        description="Fuse visual and audio into visual-style JSON via Qwen."
    )
    parser.add_argument("--visual-json", required=True)
    parser.add_argument("--transcript-json", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--model",
        default="qwen2.5:latest",
        help="Ollama model tag (run `ollama list` to see installed names)",
    )
    parser.add_argument("--ollama-url", default="http://localhost:11434/api/generate")

    args = parser.parse_args()

    visual_json = load_json(args.visual_json)
    transcript_json = load_json(args.transcript_json)
    prompt = load_text(args.prompt)

    response_text = ""
    try:
        response_text = call_qwen(
            visual_json=visual_json,
            transcript_json=transcript_json,
            prompt=prompt,
            model=args.model,
            ollama_url=args.ollama_url,
        )
        llm_output = parse_llm_json(response_text)
    except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
        if response_text:
            debug_path = Path(args.output).with_name(
                Path(args.output).stem + "_raw_llm.txt"
            )
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            debug_path.write_text(response_text, encoding="utf-8")
            print(f"Raw model output saved to {debug_path}")
        raise exc
    llm_output = normalize_legacy_output(llm_output)
    llm_output = validate_objects(llm_output, visual_json, transcript_json)
    llm_output = reconcile_person_fusion(llm_output, visual_json, transcript_json)
    llm_output = reconcile_communications(llm_output, transcript_json)
    llm_output = reconcile_counts_by_class(llm_output, visual_json)
    llm_output = add_metadata(
        llm_output=llm_output,
        visual_json_path=args.visual_json,
        transcript_json_path=args.transcript_json,
        model=args.model,
    )

    save_json(llm_output, args.output)
    print(f"LLM output saved to {args.output}")


if __name__ == "__main__":
    main()
