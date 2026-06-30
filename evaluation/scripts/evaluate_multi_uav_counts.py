#!/usr/bin/env python3
"""
Multi-UAV object count mean error vs ground-truth counts (relative: (pred - gt) / gt).

Prediction: counts_after_match in fusion/cross_view.json
             (unique objects = matched pairs merged + unmatched detections).
Ground truth: evaluation/groundTruth_data/visual/objects_count.xlsx sheet 'per_scenario'.

Usage (from repo root):
  python evaluation/scripts/evaluate_multi_uav_counts.py
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from eval_common import (  # noqa: E402
    COUNT_CLASS_NAMES,
    DEFAULT_MANIFEST,
    GT_OBJECTS_COUNT_XLSX,
    MULTI_UAV_COUNT_REPORT_XLSX,
    REPO_ROOT,
    build_img_metadata,
    cross_view_result_for_run,
    fused_counts_from_cross_view,
    iter_multi_uav_runs,
    load_class_names,
    load_scenario_count_gt,
    MEAN_RELATIVE_COUNT_ERROR_COL,
    mean_relative_count_error,
    relative_count_error,
    sorted_lightings,
    write_excel,
)


def evaluate_multi_uav_counts(*, manifest_path: Path, report_path: Path) -> Path:
    class_names = load_class_names()
    img_lighting = build_img_metadata()
    gt_counts = load_scenario_count_gt()
    # eval class id order: 0 person, 1 vehicle, 2 emergency_vehicle
    class_id_by_name = {COUNT_CLASS_NAMES[i]: i for i in range(len(COUNT_CLASS_NAMES))}

    per_run_rows: List[dict] = []
    count_rows_by_light_cls: Dict[Tuple[str, int], List[dict]] = defaultdict(list)
    n_runs = 0
    n_with_counts = 0

    for run in iter_multi_uav_runs(manifest_path):
        n_runs += 1
        if run.scenario_num not in gt_counts:
            continue
        lighting = img_lighting.get(run.img_ref, "unknown")
        payload = cross_view_result_for_run(run)
        pred_counts = fused_counts_from_cross_view(payload)
        if pred_counts is None:
            continue
        n_with_counts += 1
        gt_scene = gt_counts[run.scenario_num]

        for name in COUNT_CLASS_NAMES:
            gt_c = gt_scene.get(name, 0)
            pred_c = pred_counts.get(name, 0)
            err = pred_c - gt_c
            cls_id = class_id_by_name[name]
            row = {
                    "run_id": run.run_id,
                    "scenario": f"scenario{run.scenario_num}",
                    "img_ref": f"img{run.img_ref}",
                    "views": f"{run.view_ref_id}+{run.view_other_id}",
                    "lighting": lighting,
                    "same_incident": bool(payload.get("same_incident")),
                    "class": class_names[cls_id],
                    "gt_count": gt_c,
                    "pred_count": pred_c,
                    "count_error": err,
                    "relative_count_error": relative_count_error(pred_c, gt_c),
                    "cross_view_source": (
                        str(run.cross_view_json.relative_to(REPO_ROOT))
                        if run.cross_view_json
                        else "(computed)"
                    ),
                }
            count_rows_by_light_cls[(lighting, cls_id)].append(row)
            per_run_rows.append(row)

    lightings = sorted_lightings(img_lighting, list(img_lighting.keys()))
    summary_rows: List[dict] = []
    for lighting in lightings + ["all"]:
        for cls_id, name in enumerate(COUNT_CLASS_NAMES):
            keys = [(l, cls_id) for l in lightings] if lighting == "all" else [(lighting, cls_id)]
            rows: List[dict] = []
            for k in keys:
                rows.extend(count_rows_by_light_cls[k])
            summary_rows.append(
                {
                    "lighting": lighting,
                    "class": class_names[cls_id],
                    "n_runs": len(rows),
                    MEAN_RELATIVE_COUNT_ERROR_COL: mean_relative_count_error(rows),
                }
            )

    written = write_excel(
        report_path,
        {
            "by_lighting_class": pd.DataFrame(summary_rows),
            "per_run": pd.DataFrame(per_run_rows),
        },
    )
    df_summary = pd.DataFrame(summary_rows)
    print(f"Multi-UAV runs scanned: {n_runs} ({n_with_counts} with counts_after_match)")
    print(f"GT counts: {GT_OBJECTS_COUNT_XLSX} sheet 'per_scenario'")
    print(f"Wrote {written}")
    if not df_summary.empty:
        print("Relative mean count error ((pred - gt) / gt; 0 = exact):")
        print(df_summary.to_string(index=False))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-UAV count mean error from cross_view counts_after_match."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=MULTI_UAV_COUNT_REPORT_XLSX)
    args = parser.parse_args()
    evaluate_multi_uav_counts(manifest_path=args.manifest, report_path=args.output)


if __name__ == "__main__":
    main()
