#!/usr/bin/env python3
"""
Check whether fusion/sls.json objects match communication_demand.py policy.

For each object, recomputes (throughput_need, traffic_demand_mbps) using the same
inputs as sls_builder.publish_object_with_mbps (class, role, distance, audio_only,
scene service_types, has_fire, thermal_imagery_consumer from llm_output by index).

Output: evaluation/results/sls_eval/communication_demand_compliance.xlsx

Usage (from repo root):
  python evaluation/scripts/evaluate_communication_demand_compliance.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
_FUSION_SLS = _REPO_ROOT / "fusion" / "sls"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_FUSION_SLS) not in sys.path:
    sys.path.insert(0, str(_FUSION_SLS))

from communication_demand import (  # noqa: E402
    PERSON_AT_RISK_METERS,
    RESPONDER_ROLES,
    ROLE_BASE_MBPS,
    SERVICE_MBPS,
    VEHICLE_OPERATIONAL_METERS,
    communication_demand_for_object,
)
from eval_common import (  # noqa: E402
    COMM_DEMAND_COMPLIANCE_REPORT_XLSX,
    DEFAULT_MANIFEST,
    REPO_ROOT,
    build_img_metadata,
    write_excel,
)

MBPS_TOLERANCE = 1e-6


def policy_reference_rows() -> List[dict]:
    return [
        {
            "rule_id": "audio_only",
            "condition": "audio_only == true",
            "throughput_need": "low",
            "traffic_demand_mbps": SERVICE_MBPS["voice"],
        },
        {
            "rule_id": "thermal_consumer_stream",
            "condition": "thermal_imagery_consumer + scene has video/image_or_video",
            "throughput_need": "high",
            "traffic_demand_mbps": SERVICE_MBPS["stream_visual"],
        },
        {
            "rule_id": "thermal_consumer_point",
            "condition": "thermal_imagery_consumer + scene has thermal_image/image_transfer",
            "throughput_need": "high",
            "traffic_demand_mbps": SERVICE_MBPS["point_visual"],
        },
        {
            "rule_id": "emergency_vehicle",
            "condition": "class == emergency_vehicle",
            "throughput_need": "medium",
            "traffic_demand_mbps": ROLE_BASE_MBPS["emergency_vehicle_medium"],
        },
        {
            "rule_id": "person_responder",
            "condition": f"person + role in {sorted(RESPONDER_ROLES)}",
            "throughput_need": "medium",
            "traffic_demand_mbps": ROLE_BASE_MBPS["medium"],
        },
        {
            "rule_id": "person_at_risk",
            "condition": f"person + has_fire + distance <= {PERSON_AT_RISK_METERS}m",
            "throughput_need": "medium",
            "traffic_demand_mbps": ROLE_BASE_MBPS["at_risk_no_service"],
        },
        {
            "rule_id": "person_default",
            "condition": "person (other)",
            "throughput_need": "low",
            "traffic_demand_mbps": ROLE_BASE_MBPS["low"],
        },
        {
            "rule_id": "vehicle_near_fire",
            "condition": f"normal_vehicle + distance <= {VEHICLE_OPERATIONAL_METERS}m",
            "throughput_need": "medium",
            "traffic_demand_mbps": ROLE_BASE_MBPS["medium"],
        },
        {
            "rule_id": "vehicle_default",
            "condition": "normal_vehicle (far)",
            "throughput_need": "low",
            "traffic_demand_mbps": ROLE_BASE_MBPS["low"],
        },
        {
            "rule_id": "default",
            "condition": "other class",
            "throughput_need": "low",
            "traffic_demand_mbps": ROLE_BASE_MBPS["low"],
        },
    ]


def scene_service_types(sls: Dict[str, Any]) -> List[str]:
    comms = sls.get("communications") or {}
    types = comms.get("service_types", comms.get("service_type", ["voice"]))
    if not isinstance(types, list):
        types = [types]
    return types or ["voice"]


def profile_from_sls_object(obj: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "class": obj.get("class"),
        "distance_to_fire": obj.get("distance_to_fire"),
        "inferred_role": obj.get("inferred_role"),
        "audio_only": obj.get("audio_only"),
    }


def load_llm_objects(run_dir: Path) -> Optional[List[Dict[str, Any]]]:
    llm_path = run_dir / "fusion" / "llm_output.json"
    if not llm_path.is_file():
        return None
    data = json.loads(llm_path.read_text(encoding="utf-8"))
    return data.get("objects") or []


def sls_files_for_run(run_dir: Path) -> List[Path]:
    fusion = run_dir / "fusion"
    paths: List[Path] = []
    main = fusion / "sls.json"
    if main.is_file():
        paths.append(main)
    for path in sorted(fusion.glob("sls_*.json")):
        if path not in paths:
            paths.append(path)
    return paths


def mbps_equal(a: Any, b: float) -> bool:
    try:
        return abs(float(a) - b) <= MBPS_TOLERANCE
    except (TypeError, ValueError):
        return False


def evaluate_sls_file(
    sls_path: Path,
    *,
    run_id: str,
    config: str,
    llm_objects: Optional[List[Dict[str, Any]]],
    lighting: str,
) -> List[dict]:
    sls = json.loads(sls_path.read_text(encoding="utf-8"))
    services = scene_service_types(sls)
    has_fire = bool(sls.get("has_fire"))
    rows: List[dict] = []
    rel_sls = str(sls_path.relative_to(REPO_ROOT))

    for idx, obj in enumerate(sls.get("objects") or []):
        thermal = False
        if llm_objects is not None and idx < len(llm_objects):
            thermal = bool(llm_objects[idx].get("thermal_imagery_consumer"))
        profile = profile_from_sls_object(obj)
        exp_need, exp_mbps = communication_demand_for_object(
            profile,
            services,
            thermal_consumer=thermal,
            has_fire=has_fire,
        )
        act_need = obj.get("throughput_need")
        act_mbps = obj.get("traffic_demand_mbps")
        need_ok = act_need == exp_need
        mbps_ok = mbps_equal(act_mbps, exp_mbps)
        rows.append(
            {
                "config": config,
                "run_id": run_id,
                "lighting": lighting,
                "sls_file": rel_sls,
                "object_index": idx,
                "class": obj.get("class"),
                "audio_only": bool(obj.get("audio_only")),
                "inferred_role": obj.get("inferred_role"),
                "distance_to_fire": obj.get("distance_to_fire"),
                "thermal_imagery_consumer": thermal,
                "scene_service_types": ",".join(services),
                "has_fire": has_fire,
                "expected_throughput_need": exp_need,
                "actual_throughput_need": act_need,
                "throughput_need_ok": need_ok,
                "expected_traffic_demand_mbps": exp_mbps,
                "actual_traffic_demand_mbps": act_mbps,
                "traffic_mbps_ok": mbps_ok,
                "fully_compliant": need_ok and mbps_ok,
            }
        )
    return rows


def iter_manifest_run_dirs(manifest_path: Path) -> List[Tuple[str, str, Path]]:
    out: List[Tuple[str, str, Path]] = []
    if not manifest_path.is_file():
        return out
    with open(manifest_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rel = str(row.get("run_dir", "")).strip()
            if not rel:
                continue
            run_dir = REPO_ROOT / rel.replace("\\", "/")
            if not run_dir.is_dir():
                continue
            multi = str(row.get("multi_uav", "")).lower() in ("true", "1", "yes")
            out.append(
                (
                    "multi" if multi else "mono",
                    str(row.get("run_id", run_dir.name)),
                    run_dir,
                )
            )
    return out


def summarize_by_run(per_object: List[dict]) -> pd.DataFrame:
    if not per_object:
        return pd.DataFrame()
    df = pd.DataFrame(per_object)
    by_run = df.groupby(["config", "run_id"], dropna=False).agg(
        n_objects=("object_index", "count"),
        throughput_need_accuracy=("throughput_need_ok", "mean"),
        traffic_mbps_accuracy=("traffic_mbps_ok", "mean"),
        full_compliance_rate=("fully_compliant", "mean"),
        n_need_mismatch=("throughput_need_ok", lambda s: int((~s).sum())),
        n_mbps_mismatch=("traffic_mbps_ok", lambda s: int((~s).sum())),
    )
    by_run = by_run.reset_index()
    for col in (
        "throughput_need_accuracy",
        "traffic_mbps_accuracy",
        "full_compliance_rate",
    ):
        by_run[col] = by_run[col].round(4)
    return by_run


def evaluate_compliance(
    *,
    manifest_path: Path,
    report_path: Path,
) -> Path:
    img_lighting = build_img_metadata()
    per_object: List[dict] = []
    n_runs = 0
    n_sls_files = 0

    for config, run_id, run_dir in iter_manifest_run_dirs(manifest_path):
        sls_paths = sls_files_for_run(run_dir)
        if not sls_paths:
            continue
        n_runs += 1
        llm_objects = load_llm_objects(run_dir)
        lighting = "unknown"
        for img_num, light in img_lighting.items():
            if f"img{img_num}" in run_id:
                lighting = light
                break

        for sls_path in sls_paths:
            n_sls_files += 1
            per_object.extend(
                evaluate_sls_file(
                    sls_path,
                    run_id=run_id,
                    config=config,
                    llm_objects=llm_objects,
                    lighting=lighting,
                )
            )

    policy_df = pd.DataFrame(policy_reference_rows())
    if not per_object:
        empty = pd.DataFrame(
            columns=[
                "config",
                "run_id",
                "fully_compliant",
                "traffic_mbps_ok",
                "throughput_need_ok",
            ]
        )
        return write_excel(
            report_path,
            {
                "policy_rules": policy_df,
                "metrics_by_run": empty,
                "per_object": empty,
            },
        )

    per_object_df = pd.DataFrame(per_object)
    by_run_df = summarize_by_run(per_object)

    global_metrics = pd.DataFrame(
        [
            {
                "metric": "objects_evaluated",
                "value": len(per_object_df),
            },
            {
                "metric": "runs_with_sls",
                "value": n_runs,
            },
            {
                "metric": "sls_files",
                "value": n_sls_files,
            },
            {
                "metric": "throughput_need_compliance_rate",
                "value": round(per_object_df["throughput_need_ok"].mean(), 4),
            },
            {
                "metric": "traffic_mbps_compliance_rate",
                "value": round(per_object_df["traffic_mbps_ok"].mean(), 4),
            },
            {
                "metric": "full_compliance_rate",
                "value": round(per_object_df["fully_compliant"].mean(), 4),
            },
        ]
    )

    by_class = (
        per_object_df.groupby("class", dropna=False)
        .agg(
            n=("fully_compliant", "count"),
            throughput_need_accuracy=("throughput_need_ok", "mean"),
            traffic_mbps_accuracy=("traffic_mbps_ok", "mean"),
            full_compliance_rate=("fully_compliant", "mean"),
        )
        .reset_index()
    )
    for col in (
        "throughput_need_accuracy",
        "traffic_mbps_accuracy",
        "full_compliance_rate",
    ):
        by_class[col] = by_class[col].round(4)

    written = write_excel(
        report_path,
        {
            "policy_rules": policy_df,
            "metrics_global": global_metrics,
            "metrics_by_run": by_run_df,
            "metrics_by_class": by_class,
            "per_object": per_object_df,
        },
    )
    print(f"Runs with SLS: {n_runs}, SLS files: {n_sls_files}, objects: {len(per_object_df)}")
    print(global_metrics.to_string(index=False))
    print(f"Wrote {written}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SLS compliance vs fusion/sls/communication_demand.py"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=COMM_DEMAND_COMPLIANCE_REPORT_XLSX)
    args = parser.parse_args()
    evaluate_compliance(manifest_path=args.manifest, report_path=args.output)


if __name__ == "__main__":
    main()
