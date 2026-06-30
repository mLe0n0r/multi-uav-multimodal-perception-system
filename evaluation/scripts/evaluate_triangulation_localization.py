#!/usr/bin/env python3
"""
Localization via cross-view triangulation (multi-UAV).

Only for algorithm matches that exactly match a row in objects_matches.xlsx.
Prediction:  matches[].position in fusion/cross_view.json (triangulated XY).
Reference:   objects_position.xlsx at view_a / local_id_a for that GT row.

Reports MAE error in meters (no RMSE). Use per_correct_match for outliers.

Usage (from repo root):
  python evaluation/scripts/evaluate_triangulation_localization.py
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
    DEFAULT_MANIFEST,
    EVAL_CLASS_IDS,
    EXCEL_CLASS_TO_EVAL_ID,
    GT_MATCHES_XLSX,
    GT_POSITIONS_XLSX,
    LIGHTING_EN,
    TRIANGULATION_LOCALIZATION_REPORT_XLSX,
    build_cross_view_preds_by_pair,
    build_gt_matches_by_pair,
    cross_view_entity_from_match,
    gt_xy_for_matched_entity,
    gt_xy_spread_between_views,
    load_class_names,
    load_gt_matches_df,
    load_objects_position_xlsx,
    localization_errors,
    object_match_entity_ids,
    pair_key_from_row,
    summarize_localization_metrics,
    write_excel,
    xy_from_position_field,
)


def evaluate_triangulation_localization(
    *, manifest_path: Path, gt_matches_path: Path, report_path: Path
) -> Path:
    class_names = load_class_names()
    positions_df = load_objects_position_xlsx()
    gt_df = load_gt_matches_df(gt_matches_path)
    gt_by_pair = build_gt_matches_by_pair(gt_df)
    pred_by_pair = build_cross_view_preds_by_pair(manifest_path)

    class_by_entity: Dict[Tuple[str, str, str, str, int, int], str] = {}
    for _, row in gt_df.iterrows():
        pk = pair_key_from_row(
            str(row["scenario"]),
            str(row["group"]),
            str(row["view_a"]),
            str(row["view_b"]),
        )
        ent = object_match_entity_ids(
            str(row["view_a"]),
            int(row["local_id_a"]),
            str(row["view_b"]),
            int(row["local_id_b"]),
        )
        va = pk[2]
        rows = positions_df[
            (positions_df["img"].astype(str) == va)
            & (positions_df["obj_id"] == ent[0])
        ]
        if not rows.empty:
            class_by_entity[(*pk, ent[0], ent[1])] = str(rows.iloc[0]["classe"]).strip()

    pair_rows: List[dict] = []
    loc_err_x_by_light_cls: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    loc_err_y_by_light_cls: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    loc_err_pos_by_light_cls: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    n_correct_matches = 0
    n_skipped_no_tri = 0
    n_skipped_no_gt_xy = 0

    for pk, gt_set in sorted(gt_by_pair.items()):
        scenario, group, view_a, view_b = pk
        pred_entry = pred_by_pair.get(pk, {})
        pred_matches = pred_entry.get("matches", [])
        lighting = LIGHTING_EN.get(group, group)

        for m in pred_matches:
            ent = cross_view_entity_from_match(m)
            if ent not in gt_set:
                continue

            tri_xy = xy_from_position_field(m.get("position"))
            if tri_xy is None:
                n_skipped_no_tri += 1
                continue

            g_pos = gt_xy_for_matched_entity(
                positions_df,
                view_a,
                ent[0],
                view_b,
                ent[1],
            )
            if g_pos is None:
                n_skipped_no_gt_xy += 1
                continue

            n_correct_matches += 1
            err_x, err_y, err_pos = localization_errors(g_pos, tri_xy)
            spread = gt_xy_spread_between_views(
                positions_df, view_a, ent[0], view_b, ent[1]
            )

            cls_name = class_by_entity.get((*pk, ent[0], ent[1]), "")
            cls_id = EXCEL_CLASS_TO_EVAL_ID.get(cls_name, -1)

            if cls_id in EVAL_CLASS_IDS:
                loc_err_x_by_light_cls[(lighting, cls_id)].append(err_x)
                loc_err_y_by_light_cls[(lighting, cls_id)].append(err_y)
                loc_err_pos_by_light_cls[(lighting, cls_id)].append(err_pos)

            pair_rows.append(
                {
                    "scenario": scenario,
                    "group": group,
                    "lighting": lighting,
                    "view_a": view_a,
                    "view_b": view_b,
                    "local_id_a": ent[0],
                    "local_id_b": ent[1],
                    "class": cls_name,
                    "gt_x_m": round(g_pos[0], 4),
                    "gt_y_m": round(g_pos[1], 4),
                    "pred_x_m": round(tri_xy[0], 4),
                    "pred_y_m": round(tri_xy[1], 4),
                    "err_x_m": round(err_x, 4),
                    "err_y_m": round(err_y, 4),
                    "err_pos_m": round(err_pos, 4),
                    "gt_spread_ab_m": round(spread, 4) if spread is not None else None,
                    "run_id": pred_entry.get("run_id", ""),
                }
            )

    lightings = sorted({LIGHTING_EN.get(g, g) for g in gt_df["group"].astype(str)} | {"all"})
    summary_rows: List[dict] = []
    for lighting in lightings:
        for cls_id in EVAL_CLASS_IDS:
            keys = (
                [(l, cls_id) for l in lightings if l != "all"]
                if lighting == "all"
                else [(lighting, cls_id)]
            )
            errs_x: List[float] = []
            errs_y: List[float] = []
            errs_pos: List[float] = []
            for k in keys:
                if k[0] == "all":
                    continue
                errs_x.extend(loc_err_x_by_light_cls[k])
                errs_y.extend(loc_err_y_by_light_cls[k])
                errs_pos.extend(loc_err_pos_by_light_cls[k])
            row = summarize_localization_metrics(errs_x, errs_y, errs_pos)
            summary_rows.append(
                {
                    "lighting": lighting,
                    "class": class_names[cls_id],
                    **row,
                }
            )

    written = write_excel(
        report_path,
        {
            "by_lighting_class": pd.DataFrame(summary_rows),
            "per_correct_match": pd.DataFrame(pair_rows),
        },
    )
    df_summary = pd.DataFrame(summary_rows)

    print(f"GT matches: {gt_matches_path} ({len(gt_df)} rows)")
    print(
        f"Correct matches with triangulation: {n_correct_matches} "
        f"(skipped: {n_skipped_no_tri} no position, {n_skipped_no_gt_xy} no GT xy)"
    )
    print(f"Wrote {written}")
    if not df_summary.empty:
        show = df_summary[df_summary["n_pairs"].fillna(0).astype(int) > 0]
        print("Triangulation error vs objects_position (MAE, meters):")
        print(
            show[["lighting", "class", "n_pairs", "mae_pos_m"]].to_string(index=False)
        )
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Triangulation localization for GT-confirmed correct matches only."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--gt-matches", type=Path, default=GT_MATCHES_XLSX)
    parser.add_argument("--output", type=Path, default=TRIANGULATION_LOCALIZATION_REPORT_XLSX)
    args = parser.parse_args()
    evaluate_triangulation_localization(
        manifest_path=args.manifest,
        gt_matches_path=args.gt_matches,
        report_path=args.output,
    )


if __name__ == "__main__":
    main()
