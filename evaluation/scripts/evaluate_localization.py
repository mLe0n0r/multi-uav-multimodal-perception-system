#!/usr/bin/env python3
"""
Localization evaluation: MAE_x / MAE_y / MAE_pos by object id (meters).

Ground-truth positions: evaluation/groundTruth_data/visual/objects_position.xlsx
Predictions: position.x / position.y from system visual.json (same obj_id).

Usage (from repo root):
  python evaluation/scripts/evaluate_localization.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from eval_common import (  # noqa: E402
    DEFAULT_MANIFEST,
    EVAL_CLASS_IDS,
    EXCEL_CLASS_TO_EVAL_ID,
    GT_POSITIONS_XLSX,
    LOCALIZATION_REPORT_XLSX,
    REPO_ROOT,
    load_eval_context,
    load_objects_position_xlsx,
    localization_errors,
    pred_xy_by_object_id,
    sorted_lightings,
    summarize_localization_metrics,
    write_excel,
)


def evaluate_localization(*, manifest_path: Path, report_path: Path) -> Path:
    ctx = load_eval_context(manifest_path)
    positions_df = load_objects_position_xlsx()

    loc_err_x_by_light_cls: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    loc_err_y_by_light_cls: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    loc_err_pos_by_light_cls: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    pair_rows: List[dict] = []

    for img_num in ctx.image_ids:
        lighting = ctx.img_lighting.get(img_num, "unknown")
        img_key = f"img{img_num}"
        img_gt = positions_df[positions_df["img"].astype(str) == img_key]
        if img_gt.empty:
            continue

        pred_path = ctx.pred_paths[img_num]
        pred_payload = json.loads(pred_path.read_text(encoding="utf-8"))
        pred_xy_by_id = pred_xy_by_object_id(pred_payload)

        for _, erow in img_gt.iterrows():
            obj_id = int(erow["obj_id"])
            p_pos = pred_xy_by_id.get(obj_id)
            if p_pos is None:
                continue
            cls_id = EXCEL_CLASS_TO_EVAL_ID.get(str(erow["classe"]).strip())
            if cls_id is None or cls_id not in EVAL_CLASS_IDS:
                continue

            g_pos = (float(erow["x (m)"]), float(erow["y (m)"]))
            err_x, err_y, err_pos = localization_errors(g_pos, p_pos)
            key = (lighting, cls_id)
            loc_err_x_by_light_cls[key].append(err_x)
            loc_err_y_by_light_cls[key].append(err_y)
            loc_err_pos_by_light_cls[key].append(err_pos)
            pair_rows.append(
                {
                    "img": img_key,
                    "obj_id": obj_id,
                    "lighting": lighting,
                    "class": ctx.class_names[cls_id],
                    "gt_x_m": round(g_pos[0], 4),
                    "gt_y_m": round(g_pos[1], 4),
                    "pred_x_m": round(p_pos[0], 4),
                    "pred_y_m": round(p_pos[1], 4),
                    "err_x_m": round(err_x, 4),
                    "err_y_m": round(err_y, 4),
                    "err_pos_m": round(err_pos, 4),
                    "prediction_source": str(pred_path.relative_to(REPO_ROOT)),
                }
            )

    lightings = sorted_lightings(ctx.img_lighting, ctx.image_ids)
    summary_rows: List[dict] = []
    for lighting in lightings + ["all"]:
        for cls_id in EVAL_CLASS_IDS:
            keys = [(l, cls_id) for l in lightings] if lighting == "all" else [(lighting, cls_id)]
            errs_x: List[float] = []
            errs_y: List[float] = []
            errs_pos: List[float] = []
            for k in keys:
                errs_x.extend(loc_err_x_by_light_cls[k])
                errs_y.extend(loc_err_y_by_light_cls[k])
                errs_pos.extend(loc_err_pos_by_light_cls[k])
            row = summarize_localization_metrics(errs_x, errs_y, errs_pos)
            summary_rows.append(
                {
                    "lighting": lighting,
                    "class": ctx.class_names[cls_id],
                    **row,
                }
            )

    written = write_excel(
        report_path,
        {
            "by_lighting_class": pd.DataFrame(summary_rows),
            "per_pair": pd.DataFrame(pair_rows),
        },
    )
    df_summary = pd.DataFrame(summary_rows)
    print(f"GT positions: {GT_POSITIONS_XLSX}")
    print(f"Wrote {written}")
    if not df_summary.empty:
        print("Localization (by obj_id, meters):")
        cols = [c for c in ("lighting", "class", "n_pairs", "mae_x_m", "mae_y_m", "mae_pos_m") if c in df_summary.columns]
        print(df_summary[cols].to_string(index=False))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Localization MAE vs objects_position.xlsx (matched by obj_id)."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=LOCALIZATION_REPORT_XLSX)
    args = parser.parse_args()
    evaluate_localization(manifest_path=args.manifest, report_path=args.output)


if __name__ == "__main__":
    main()
