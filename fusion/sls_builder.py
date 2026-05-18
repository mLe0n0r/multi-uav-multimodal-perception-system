import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

VALID_CLASSES = ("person", "normal_vehicle", "emergency_vehicle")

# Scene communications only — no per-modality mention counts in the final SLS.
COMMUNICATIONS_KEYS = (
    "speaker_count",
    "key_communications",
    "service_types",
    "service_inference_basis",
)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def assign_traffic_demand(obj: Dict[str, Any], scene_service_types: List[str]) -> float:
    need = obj.get("throughput_need", "low")
    services = scene_service_types or ["voice"]

    if need == "high" and "command_aggregation" in services:
        return 10.0
    if need == "high" and "thermal_image" in services:
        return 7.0
    if need == "high" and ("video" in services or "image_or_video" in services):
        return 5.0
    if need == "high":
        return 3.0
    if need == "medium" and ("image_transfer" in services or "image_or_video" in services):
        return 2.0
    if need == "medium":
        return 1.0
    if need == "low" and "basic_data" in services:
        return 0.5
    return 0.2


def apply_traffic(obj: Dict[str, Any], scene_service_types: List[str]) -> float:
    traffic = assign_traffic_demand(obj, scene_service_types)
    if obj.get("audio_only"):
        traffic = round(traffic * 0.5, 2)
    return traffic


def default_semantic_for_visual(visual_obj: Dict[str, Any]) -> Dict[str, Any]:
    """Fallback when the LLM omitted a visually detected object."""
    cls = visual_obj.get("class", "person")
    role_by_class = {
        "person": "unknown_person",
        "normal_vehicle": "background_vehicle",
        "emergency_vehicle": "emergency_vehicle",
    }
    return {
        "inferred_role": role_by_class.get(cls, "unknown_person"),
        "role_inference_basis": ["visual_detection"],
        "role_confidence": None,
        "risk_level": "medium",
        "throughput_need": "low",
    }


def index_llm_objects(
    llm_output: Dict[str, Any],
) -> tuple[Dict[int, Dict[str, Any]], List[Dict[str, Any]]]:
    by_id: Dict[int, Dict[str, Any]] = {}
    audio_only: List[Dict[str, Any]] = []
    for obj in llm_output.get("objects", []):
        oid = obj.get("id")
        if oid is not None:
            by_id[oid] = obj
        elif obj.get("audio_only"):
            audio_only.append(obj)
    return by_id, audio_only


def build_visual_object_entry(
    semantic: Dict[str, Any],
    visual_obj: Dict[str, Any],
    scene_service_types: List[str],
) -> Dict[str, Any]:
    risk = semantic.get("risk_level") or "medium"
    need = semantic.get("throughput_need") or "low"
    traffic_input = {**semantic, "risk_level": risk, "throughput_need": need}

    entry = {
        "id": visual_obj["id"],
        "class": visual_obj.get("class", semantic.get("class")),
        "detection_confidence": visual_obj.get("detection_confidence"),
        "position": visual_obj.get("position"),
        "localization_confidence": visual_obj.get("localization_confidence"),
        "distance_to_fire": visual_obj.get("distance_to_fire"),
        "inferred_role": semantic.get("inferred_role"),
        "role_inference_basis": semantic.get("role_inference_basis", []),
        "role_confidence": semantic.get("role_confidence"),
        "risk_level": risk,
        "throughput_need": need,
        "traffic_demand_mbps": apply_traffic(traffic_input, scene_service_types),
    }
    return entry


def build_audio_only_object_entry(
    semantic: Dict[str, Any], scene_service_types: List[str]
) -> Dict[str, Any]:
    entry = {
        "id": None,
        "class": semantic.get("class"),
        "audio_only": True,
        "reason": semantic.get("reason", ""),
        "inferred_role": semantic.get("inferred_role"),
        "risk_level": semantic.get("risk_level"),
        "throughput_need": semantic.get("throughput_need"),
        "traffic_demand_mbps": apply_traffic(semantic, scene_service_types),
    }
    return entry


def normalize_legacy_llm(llm_output: Dict[str, Any]) -> Dict[str, Any]:
    if llm_output.get("objects"):
        return llm_output

    incident = llm_output.get("incident_summary", {}) or {}
    llm_output.setdefault("has_fire", incident.get("has_fire"))
    llm_output.setdefault("scenario_priority", incident.get("scenario_priority"))
    llm_output.setdefault("summary", incident.get("summary"))
    llm_output.setdefault("communications", incident.get("audio_context", {}))

    objects: List[Dict[str, Any]] = []
    for entity in llm_output.get("entities", []):
        objects.append(
            {
                "id": entity.get("source_object_id", entity.get("id")),
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
                    "risk_level": item.get("risk_level"),
                    "throughput_need": item.get("throughput_need"),
                }
            )
    llm_output["objects"] = objects
    return llm_output


def counts_from_objects(objects: List[Dict[str, Any]]) -> Dict[str, int]:
    """Final fused counts: one integer per class, derived from objects[]."""
    tallies = Counter(
        obj["class"] for obj in objects if obj.get("class") in VALID_CLASSES
    )
    return {cls: int(tallies.get(cls, 0)) for cls in VALID_CLASSES}


def communications_for_sls(llm_output: Dict[str, Any], service_types: List[str]) -> Dict[str, Any]:
    raw = llm_output.get("communications", {}) or {}
    comms = {key: raw[key] for key in COMMUNICATIONS_KEYS if key in raw}
    comms["service_types"] = service_types
    comms.setdefault("key_communications", [])
    comms.setdefault("service_inference_basis", [])
    return comms


def scene_service_types(llm_output: Dict[str, Any]) -> List[str]:
    comms = llm_output.get("communications", {}) or {}
    types = comms.get("service_types", comms.get("service_type", ["voice"]))
    if not isinstance(types, list):
        types = [types]
    return types or ["voice"]


def build_sls(llm_output: Dict[str, Any], visual_json: Dict[str, Any]) -> Dict[str, Any]:
    llm_output = normalize_legacy_llm(llm_output)
    services = scene_service_types(llm_output)
    semantic_by_id, audio_only_list = index_llm_objects(llm_output)

    objects_out: List[Dict[str, Any]] = []

    # Every visually detected object must appear in the final SLS.
    for visual_obj in visual_json.get("objects", []):
        oid = visual_obj.get("id")
        if oid is None:
            continue
        semantic = semantic_by_id.get(oid) or default_semantic_for_visual(visual_obj)
        objects_out.append(build_visual_object_entry(semantic, visual_obj, services))

    for semantic in audio_only_list:
        objects_out.append(build_audio_only_object_entry(semantic, services))

    return {
        "has_fire": llm_output.get("has_fire", visual_json.get("has_fire")),
        "scenario_priority": llm_output.get("scenario_priority", "unknown"),
        "summary": llm_output.get("summary", ""),
        "camera": visual_json.get("camera"),
        "counts_by_class": counts_from_objects(objects_out),
        "communications": communications_for_sls(llm_output, services),
        "objects": objects_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build final SLS (visual JSON schema + communications + traffic)."
    )
    parser.add_argument("--llm-output", required=True)
    parser.add_argument("--visual-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    llm_output = load_json(args.llm_output)
    visual_json = load_json(args.visual_json)
    sls = build_sls(llm_output, visual_json)
    save_json(sls, args.output)
    print(f"SLS saved to {args.output}")


if __name__ == "__main__":
    main()
