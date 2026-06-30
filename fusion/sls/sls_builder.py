"""Build final SLS JSON from llm_output.json — same content plus traffic_demand_mbps."""

import argparse
import copy
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_SLS_DIR = Path(__file__).resolve().parent
_FUSION_ROOT = _SLS_DIR.parent
_LLM_CODE = _FUSION_ROOT / "llm" / "code"
for path in (_FUSION_ROOT, _SLS_DIR, _LLM_CODE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from communication_demand import communication_demand_for_object
from fused_counts import (
    VALID_CLASSES,
    deduped_visual_counts,
    fused_entity_groups,
    pick_representative_visual,
    reference_view_id,
    visual_object_lookup,
)
from llm_orchestrator import normalize_role_inference_basis


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


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


def infer_risk_level(visual_obj: Dict[str, Any], has_fire: bool) -> str:
    """Risk from proximity, class, and incident context — not from comms services."""
    cls = visual_obj.get("class", "person")
    distance = parse_distance_meters(visual_obj.get("distance_to_fire"))
    if not has_fire:
        return "low" if distance is None else "medium"

    if cls == "person":
        if distance is not None and distance <= 1.0:
            return "high"
        if distance is not None and distance <= 5.0:
            return "medium"
        return "low"

    if cls == "normal_vehicle":
        if distance is not None and distance <= 3.0:
            return "high"
        if distance is not None and distance <= 8.0:
            return "medium"
        return "low"

    if cls == "emergency_vehicle":
        return "medium"

    return "medium"


def traffic_demand_mbps_for_object(
    profile: Dict[str, Any],
    scene_service_types: List[str],
    *,
    thermal_consumer: bool = False,
    has_fire: bool = True,
) -> Tuple[str, float]:
    return communication_demand_for_object(
        profile,
        scene_service_types,
        thermal_consumer=thermal_consumer,
        has_fire=has_fire,
    )


def publish_object_with_mbps(
    obj: Dict[str, Any],
    scene_service_types: List[str],
    *,
    has_fire: bool = True,
) -> Dict[str, Any]:
    """Copy one llm_output object and attach traffic_demand_mbps (only SLS-only field)."""
    entry = {k: v for k, v in obj.items() if k not in ("id", "view_id")}
    profile = {
        "class": entry.get("class"),
        "distance_to_fire": entry.get("distance_to_fire"),
        "inferred_role": entry.get("inferred_role"),
        "audio_only": entry.get("audio_only"),
    }
    need, mbps = traffic_demand_mbps_for_object(
        profile,
        scene_service_types,
        thermal_consumer=bool(obj.get("thermal_imagery_consumer")),
        has_fire=has_fire,
    )
    entry["throughput_need"] = need
    entry["traffic_demand_mbps"] = mbps
    if entry.get("class") == "person":
        role = entry.get("inferred_role")
        if role is None or not str(role).strip():
            entry["inferred_role"] = "unknown_person"
    return entry


def default_semantic_for_visual(visual_obj: Dict[str, Any]) -> Dict[str, Any]:
    """Fallback when the LLM omitted a visually detected object."""
    cls = visual_obj.get("class", "person")
    semantic: Dict[str, Any] = {
        "risk_level": "medium",
        "throughput_need": "low",
    }
    if cls == "person":
        semantic.update(
            {
                "inferred_role": "unknown_person",
                "role_inference_basis": {
                    "source": "inference",
                    "text": "visual_detection",
                },
                "role_confidence": None,
            }
        )
    return semantic


def index_llm_objects(
    llm_output: Dict[str, Any],
    *,
    multi_view: bool = False,
) -> tuple[
    Dict[int, Dict[str, Any]],
    Dict[Tuple[str, int], Dict[str, Any]],
    Dict[int, Dict[str, Any]],
    List[Dict[str, Any]],
]:
    by_id: Dict[int, Dict[str, Any]] = {}
    by_view_key: Dict[Tuple[str, int], Dict[str, Any]] = {}
    by_list_index: Dict[int, Dict[str, Any]] = {}
    audio_only: List[Dict[str, Any]] = []
    list_index = 0
    for obj in llm_output.get("objects", []):
        if obj.get("audio_only"):
            audio_only.append(obj)
            continue
        oid = obj.get("id")
        if oid is not None:
            by_id[int(oid)] = obj
            view_id = obj.get("view_id")
            if multi_view and view_id is not None:
                by_view_key[(str(view_id), int(oid))] = obj
        else:
            by_list_index[list_index] = obj
            list_index += 1
    return by_id, by_view_key, by_list_index, audio_only


def semantic_for_fused_cluster(
    member_keys: List[Tuple[str, int]],
    by_view_key: Dict[Tuple[str, int], Dict[str, Any]],
    by_id: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    for key in member_keys:
        if key in by_view_key:
            return by_view_key[key]
    for _, oid in member_keys:
        if oid in by_id:
            return by_id[oid]
    return {}


def build_visual_object_entry(
    semantic: Dict[str, Any],
    visual_obj: Dict[str, Any],
    scene_service_types: List[str],
    has_fire: bool = True,
) -> Dict[str, Any]:
    risk = infer_risk_level(visual_obj, has_fire)
    obj_class = visual_obj.get("class", semantic.get("class"))
    thermal_consumer = bool(semantic.get("thermal_imagery_consumer"))
    profile = {
        "class": obj_class,
        "distance_to_fire": visual_obj.get("distance_to_fire"),
        "inferred_role": semantic.get("inferred_role"),
        "audio_only": semantic.get("audio_only"),
    }
    need, mbps = traffic_demand_mbps_for_object(
        profile,
        scene_service_types,
        thermal_consumer=thermal_consumer,
        has_fire=has_fire,
    )
    entry: Dict[str, Any] = {
        "class": obj_class,
        "detection_confidence": visual_obj.get("detection_confidence"),
        "position": visual_obj.get("position"),
        "localization_confidence": visual_obj.get("localization_confidence"),
        "distance_to_fire": visual_obj.get("distance_to_fire"),
        "risk_level": risk,
        "throughput_need": need,
        "traffic_demand_mbps": mbps,
    }
    if obj_class == "person":
        role = semantic.get("inferred_role")
        entry["inferred_role"] = role if role and str(role).strip() else "unknown_person"
        entry["role_inference_basis"] = normalize_role_inference_basis(
            semantic.get("role_inference_basis")
        )
        entry["role_confidence"] = semantic.get("role_confidence")
    return entry


def build_audio_only_object_entry(
    semantic: Dict[str, Any], scene_service_types: List[str]
) -> Dict[str, Any]:
    obj_class = semantic.get("class")
    profile = {**semantic, "class": obj_class, "audio_only": True}
    need, mbps = traffic_demand_mbps_for_object(profile, scene_service_types)
    entry: Dict[str, Any] = {
        "class": obj_class,
        "audio_only": True,
        "reason": semantic.get("reason", ""),
        "risk_level": semantic.get("risk_level") or "medium",
        "throughput_need": need,
        "traffic_demand_mbps": mbps,
    }
    if obj_class == "person":
        entry["inferred_role"] = semantic.get("inferred_role")
    return entry


def normalize_legacy_llm(llm_output: Dict[str, Any]) -> Dict[str, Any]:
    if llm_output.get("objects"):
        return llm_output

    incident = llm_output.get("incident_summary", {}) or {}
    llm_output.setdefault("has_fire", incident.get("has_fire"))
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


def copy_communications_from_llm(llm_output: Dict[str, Any]) -> Dict[str, Any]:
    """SLS communications must match llm_output exactly (no re-filtering)."""
    return copy.deepcopy(llm_output.get("communications") or {})


def scene_service_types(llm_output: Dict[str, Any]) -> List[str]:
    comms = llm_output.get("communications", {}) or {}
    types = comms.get("service_types", comms.get("service_type", ["voice"]))
    if not isinstance(types, list):
        types = [types]
    return types or ["voice"]


def _audio_only_counts(llm_output: Dict[str, Any]) -> Dict[str, int]:
    tallies = {cls: 0 for cls in VALID_CLASSES}
    for obj in llm_output.get("objects", []):
        if obj.get("id") is None and obj.get("audio_only"):
            cls = obj.get("class")
            if cls in tallies:
                tallies[cls] += 1
    return tallies


def fused_counts_for_sls(
    llm_output: Dict[str, Any],
    views: Optional[List[Dict[str, Any]]] = None,
    matching: Optional[Dict[str, Any]] = None,
) -> Dict[str, int]:
    audio_only = _audio_only_counts(llm_output)
    if views and len(views) > 1 and matching and matching.get("same_incident"):
        visual_counts = deduped_visual_counts(views, matching)
    elif views:
        visual_counts = deduped_visual_counts(views, None)
    else:
        visual_counts = llm_output.get("counts_by_class") or {}
    return {
        cls: int(visual_counts.get(cls, 0) or 0) + audio_only[cls] for cls in VALID_CLASSES
    }


def filter_llm_output_for_view(
    llm_output: Dict[str, Any],
    view_id: str,
    visual_json: Dict[str, Any],
) -> Dict[str, Any]:
    """Subset LLM output for one camera when same_incident is false."""
    view_id_str = str(view_id)
    visual_ids = {
        int(o["id"])
        for o in visual_json.get("objects", [])
        if o.get("id") is not None
    }
    visual_objs = [
        o for o in llm_output.get("objects", []) if not o.get("audio_only")
    ]
    use_view_tags = any(o.get("view_id") is not None for o in visual_objs)

    objects: List[Dict[str, Any]] = []
    for obj in llm_output.get("objects", []):
        if obj.get("audio_only"):
            objects.append(obj)
            continue
        oid = obj.get("id")
        if use_view_tags:
            if str(obj.get("view_id")) == view_id_str:
                objects.append(obj)
        elif oid is not None and int(oid) in visual_ids:
            tagged = copy.deepcopy(obj)
            tagged["view_id"] = view_id_str
            objects.append(tagged)

    out = copy.deepcopy(llm_output)
    out["objects"] = objects
    out["counts_by_class"] = counts_from_objects(objects)
    out["has_fire"] = bool(visual_json.get("has_fire"))
    return out


def _cleanup_stale_sls_files(run_dir: Path, *, independent: bool) -> None:
    """Remove SLS files from the other multi-view mode."""
    from run_layout import sls_path

    fusion = Path(run_dir) / "fusion"
    if independent:
        mono = sls_path(run_dir)
        if mono.is_file():
            mono.unlink()
    else:
        for path in fusion.glob("sls_*.json"):
            path.unlink()


def build_independent_view_sls_files(
    run_dir: Path,
    llm_output: Dict[str, Any],
    views: List[Dict[str, Any]],
) -> List[Path]:
    from run_layout import sls_path_for_view

    _cleanup_stale_sls_files(run_dir, independent=True)
    written: List[Path] = []
    for visual in views:
        view_id = str(visual.get("_view_id", "mono"))
        subset = filter_llm_output_for_view(llm_output, view_id, visual)
        sls = build_sls(subset, visual_json=visual)
        out_path = sls_path_for_view(run_dir, view_id)
        save_json(sls, str(out_path))
        written.append(out_path)
    return written


def build_sls(
    llm_output: Dict[str, Any],
    visual_json: Optional[Dict[str, Any]] = None,
    *,
    views: Optional[List[Dict[str, Any]]] = None,
    matching: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Final SLS = llm_output semantics/geometry + traffic_demand_mbps per object."""
    llm_output = normalize_legacy_llm(llm_output)
    services = scene_service_types(llm_output)
    view_list = views if views else ([visual_json] if visual_json else [])

    has_fire = bool(llm_output.get("has_fire"))
    if not has_fire and view_list:
        has_fire = any(v.get("has_fire") for v in view_list)
    if not has_fire and visual_json:
        has_fire = bool(visual_json.get("has_fire"))

    objects_out = [
        publish_object_with_mbps(obj, services, has_fire=has_fire)
        for obj in llm_output.get("objects", [])
    ]

    if view_list:
        counts = fused_counts_for_sls(llm_output, view_list, matching)
    else:
        counts = llm_output.get("counts_by_class") or counts_from_objects(objects_out)

    return {
        "has_fire": has_fire,
        "summary": llm_output.get("summary", ""),
        "counts_by_class": counts,
        "communications": copy_communications_from_llm(llm_output),
        "objects": objects_out,
    }


def main() -> None:
    if str(_FUSION_ROOT) not in sys.path:
        sys.path.insert(0, str(_FUSION_ROOT))
    from run_layout import (
        is_multi_view_run,
        llm_output_path,
        load_cross_view,
        load_primary_visual,
        load_visual_views,
        sls_path,
        sls_path_for_view,
        visual_views_from_cross_view,
    )

    parser = argparse.ArgumentParser(
        description="Build final SLS (visual JSON schema + communications + traffic)."
    )
    parser.add_argument(
        "--run-dir",
        help="Run folder (output/<scenario>/<run_id>); uses fusion/llm_output.json and all visuals if multi-view",
    )
    parser.add_argument("--llm-output", help="Path to llm_output.json")
    parser.add_argument("--visual-json", help="Primary visual.json for object geometry in SLS")
    parser.add_argument("--output", help="Path to sls.json (default: <run-dir>/fusion/sls.json)")
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
        llm_path = args.llm_output or str(llm_output_path(run_dir))
        out_path = args.output or str(sls_path(run_dir))
        llm_output = load_json(llm_path)
        matching = load_cross_view(run_dir)
        if is_multi_view_run(run_dir) and matching and matching.get("same_incident"):
            _cleanup_stale_sls_files(run_dir, independent=False)
            views = visual_views_from_cross_view(matching)
            if not views:
                views = load_visual_views(run_dir)
            sls = build_sls(llm_output, views=views, matching=matching)
            save_json(sls, out_path)
            print(f"SLS saved to {out_path}")
            print(f"counts_by_class: {sls.get('counts_by_class')}")
            return
        if is_multi_view_run(run_dir) and matching and not matching.get("same_incident"):
            views = visual_views_from_cross_view(matching)
            if not views:
                views = load_visual_views(run_dir)
            paths = build_independent_view_sls_files(run_dir, llm_output, views)
            print(f"Independent multi-view: {len(paths)} SLS file(s) (same_incident=false)")
            for path in paths:
                sls = load_json(str(path))
                print(f"  {path.name}: counts_by_class={sls.get('counts_by_class')}")
            return
        if args.visual_json:
            visual_json = load_json(args.visual_json)
        else:
            visual_json = load_primary_visual(run_dir)
        sls = build_sls(llm_output, visual_json)
    else:
        if not args.llm_output or not args.visual_json or not args.output:
            parser.error("Provide --run-dir or (--llm-output, --visual-json, --output)")
        llm_path = args.llm_output
        visual_path = args.visual_json
        out_path = args.output
        llm_output = load_json(llm_path)
        visual_json = load_json(visual_path)
        sls = build_sls(llm_output, visual_json)

    save_json(sls, out_path)
    print(f"SLS saved to {out_path}")
    print(f"counts_by_class: {sls.get('counts_by_class')}")


if __name__ == "__main__":
    main()
