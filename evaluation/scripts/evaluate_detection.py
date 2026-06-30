#!/usr/bin/env python3
"""
Detection evaluation: precision, recall, F1, AP@0.5 vs gt_labels.

Relative mean count error: mean (pred - gt) / gt from mono-UAV visual.json vs per_scenario GT.

Usage (from repo root):
  python evaluation/scripts/evaluate_detection.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from eval_common import (  # noqa: E402
    COUNT_CLASS_NAMES,
    DEFAULT_MANIFEST,
    DETECTION_REPORT_XLSX,
    EVAL_CLASS_IDS,
    REPO_ROOT,
    average_precision,
    boxes_from_visual_json,
    build_img_metadata,
    counts_from_visual_json,
    iter_mono_uav_runs,
    load_eval_context,
    load_scenario_count_gt,
    match_image,
    MEAN_RELATIVE_COUNT_ERROR_COL,
    mean_relative_count_error,
    parse_yolo_label_file,
    prf,
    relative_count_error,
    sorted_lightings,
    write_excel,
)


def _relative_count_mean_error(rows: List[dict]) -> Optional[float]:
    return mean_relative_count_error(rows)


def _evaluate_mono_counts(*, manifest_path: Path) -> List[dict]:
    """Mono-UAV visual.json counts vs objects_count.xlsx sheet per_scenario (one row-set per image)."""
    count_gt = load_scenario_count_gt()
    img_lighting = build_img_metadata()
    rows: List[dict] = []
    seen_images: set[int] = set()

    for run in iter_mono_uav_runs(manifest_path):
        if run.img_ref in seen_images:
            continue
        gt_scene = count_gt.get(run.scenario_num)
        if not gt_scene:
            continue
        seen_images.add(run.img_ref)
        pred_counts = counts_from_visual_json(
            json.loads(run.visual_json.read_text(encoding="utf-8"))
        )
        lighting = img_lighting.get(run.img_ref, "unknown")
        for cls_id in EVAL_CLASS_IDS:
            count_cls = COUNT_CLASS_NAMES[cls_id]
            gt_c = gt_scene.get(count_cls)
            if gt_c is None:
                continue
            pred_c = int(pred_counts.get(count_cls, 0))
            rows.append(
                {
                    "run_id": run.run_id,
                    "scenario": f"scenario{run.scenario_num}",
                    "img": f"img{run.img_ref}",
                    "lighting": lighting,
                    "class": count_cls,
                    "gt_count": int(gt_c),
                    "pred_count": pred_c,
                    "count_error": pred_c - int(gt_c),
                    "relative_count_error": relative_count_error(pred_c, int(gt_c)),
                    "prediction_source": str(run.visual_json.relative_to(REPO_ROOT)),
                }
            )
    return rows


def evaluate_detection(*, manifest_path: Path, report_path: Path) -> Path:
    ctx = load_eval_context(manifest_path)
    mono_count_rows = _evaluate_mono_counts(manifest_path=manifest_path)
    per_image_rows: List[dict] = []
    det_scores_by_light_cls: Dict[Tuple[str, int], List[Tuple[float, bool]]] = defaultdict(list)
    n_gt_by_light_cls: Dict[Tuple[str, int], int] = defaultdict(int)
    tpfpfn_by_light_cls: Dict[Tuple[str, int], List[int]] = defaultdict(lambda: [0, 0, 0])

    for img_num in ctx.image_ids:
        lighting = ctx.img_lighting.get(img_num, "unknown")
        gt_boxes = parse_yolo_label_file(ctx.gt_files[img_num])
        pred_path = ctx.pred_paths[img_num]
        pred_payload = json.loads(pred_path.read_text(encoding="utf-8"))
        pred_boxes = boxes_from_visual_json(pred_payload)

        tp, fp, fn, _scores = match_image(gt_boxes, pred_boxes)
        per_image_rows.append(
            {
                "img": f"img{img_num}",
                "lighting": lighting,
                "prediction_source": str(pred_path.relative_to(REPO_ROOT)),
                "gt_total": len(gt_boxes),
                "pred_total": len(pred_boxes),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )

        for cls_id in EVAL_CLASS_IDS:
            cls_name = ctx.class_names[cls_id]
            gt_c = sum(1 for b in gt_boxes if b.cls_id == cls_id)
            pred_c = sum(1 for b in pred_boxes if b.cls_id == cls_id)
            gt_cls = [b for b in gt_boxes if b.cls_id == cls_id]
            pred_cls = [b for b in pred_boxes if b.cls_id == cls_id]
            tpc, fpc, fnc, sc = match_image(gt_cls, pred_cls)
            key = (lighting, cls_id)
            tpfpfn_by_light_cls[key][0] += tpc
            tpfpfn_by_light_cls[key][1] += fpc
            tpfpfn_by_light_cls[key][2] += fnc
            det_scores_by_light_cls[key].extend(sc)
            n_gt_by_light_cls[key] += gt_c

            per_image_rows.append(
                {
                    "img": f"img{img_num}",
                    "lighting": lighting,
                    "class": cls_name,
                    "gt_count": gt_c,
                    "pred_count": pred_c,
                    "tp": tpc,
                    "fp": fpc,
                    "fn": fnc,
                }
            )

    lightings = sorted_lightings(ctx.img_lighting, ctx.image_ids)
    summary_rows: List[dict] = []
    for lighting in lightings + ["all"]:
        aps: List[float] = []
        for cls_id in EVAL_CLASS_IDS:
            keys = [(l, cls_id) for l in lightings] if lighting == "all" else [(lighting, cls_id)]
            tp = sum(tpfpfn_by_light_cls[k][0] for k in keys)
            fp = sum(tpfpfn_by_light_cls[k][1] for k in keys)
            fn = sum(tpfpfn_by_light_cls[k][2] for k in keys)
            n_gt = sum(n_gt_by_light_cls[k] for k in keys)
            scores: List[Tuple[float, bool]] = []
            for k in keys:
                scores.extend(det_scores_by_light_cls[k])
            precision, recall, f1 = prf(tp, fp, fn)
            ap = average_precision(scores, n_gt)
            if not np.isnan(ap):
                aps.append(ap)
            count_rows = [
                r
                for r in mono_count_rows
                if r.get("class") == COUNT_CLASS_NAMES[cls_id]
                and (lighting == "all" or r.get("lighting") == lighting)
            ]
            summary_rows.append(
                {
                    "lighting": lighting,
                    "class": ctx.class_names[cls_id],
                    "n_gt": n_gt,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "f1": round(f1, 4),
                    "ap50": round(ap, 4) if not np.isnan(ap) else None,
                    MEAN_RELATIVE_COUNT_ERROR_COL: _relative_count_mean_error(count_rows),
                }
            )
        summary_rows.append(
            {
                "lighting": lighting,
                "class": "mAP@0.5",
                "ap50": round(float(np.nanmean(aps)), 4) if aps else None,
            }
        )

    written = write_excel(
        report_path,
        {
            "by_lighting_class": pd.DataFrame(summary_rows),
            "per_image": pd.DataFrame(per_image_rows),
            "mono_count": pd.DataFrame(mono_count_rows),
        },
    )
    df_summary = pd.DataFrame(summary_rows)
    print(f"Wrote {written}")
    n_mono_images = len({r["img"] for r in mono_count_rows})
    print(
        f"Mono count: {n_mono_images} images, {len(mono_count_rows)} rows "
        f"(visual.json vs objects_count per_scenario)"
    )
    print(df_summary[df_summary["class"] != "mAP@0.5"].to_string(index=False))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Detection metrics vs ground-truth boxes.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DETECTION_REPORT_XLSX)
    args = parser.parse_args()
    evaluate_detection(manifest_path=args.manifest, report_path=args.output)


if __name__ == "__main__":
    main()
