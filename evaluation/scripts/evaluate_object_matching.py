#!/usr/bin/env python3
"""
Cross-view object matching: precision, recall, F1 vs objects_matches.xlsx.

Ground truth: evaluation/groundTruth_data/visual/objects_matches.xlsx (confirmed pairs)
Predictions:  fusion/cross_view.json (best run per image pair from manifest)

Usage (from repo root):
  python evaluation/scripts/evaluate_object_matching.py
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from eval_common import (  # noqa: E402
    DEFAULT_MANIFEST,
    GT_MATCHES_XLSX,
    LIGHTING_EN,
    OBJECT_MATCHING_REPORT_XLSX,
    build_cross_view_preds_by_pair,
    build_gt_matches_by_pair,
    cross_view_entity_from_match,
    load_gt_matches_df,
    prf,
    write_excel,
)


def evaluate_object_matching(
    *,
    manifest_path: Path,
    gt_matches_path: Path,
    report_path: Path,
) -> Path:
    gt_df = load_gt_matches_df(gt_matches_path)
    gt_by_pair = build_gt_matches_by_pair(gt_df)
    pred_by_pair = build_cross_view_preds_by_pair(manifest_path)

    pair_keys = sorted(set(gt_by_pair) | set(pred_by_pair))
    per_pair_rows: List[dict] = []
    tp_total = fp_total = fn_total = 0

    for pk in pair_keys:
        scenario, group, view_a, view_b = pk
        gt_set: Set[Tuple[int, int]] = gt_by_pair.get(pk, set())
        pred_entry = pred_by_pair.get(pk, {})
        pred_matches = pred_entry.get("matches", [])
        pred_set = {cross_view_entity_from_match(m) for m in pred_matches}

        tp = len(gt_set & pred_set)
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)
        tp_total += tp
        fp_total += fp
        fn_total += fn

        precision, recall, f1 = prf(tp, fp, fn)
        lighting = LIGHTING_EN.get(group, group)
        per_pair_rows.append(
            {
                "scenario": scenario,
                "group": group,
                "lighting": lighting,
                "view_a": view_a,
                "view_b": view_b,
                "n_gt": len(gt_set),
                "n_pred": len(pred_set),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round(precision, 4) if (tp + fp) else None,
                "recall": round(recall, 4) if (tp + fn) else None,
                "f1": round(f1, 4) if (precision + recall) else None,
                "run_id": pred_entry.get("run_id", ""),
            }
        )

    micro_p, micro_r, micro_f1 = prf(tp_total, fp_total, fn_total)

    by_lighting: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0])
    for row in per_pair_rows:
        light = row["lighting"]
        by_lighting[light][0] += row["tp"]
        by_lighting[light][1] += row["fp"]
        by_lighting[light][2] += row["fn"]

    summary_rows: List[dict] = [
        {
            "lighting": "all",
            "n_gt": sum(len(gt_by_pair.get(pk, set())) for pk in pair_keys),
            "n_pred": sum(
                len({cross_view_entity_from_match(m) for m in pred_by_pair.get(pk, {}).get("matches", [])})
                for pk in pair_keys
            ),
            "tp": tp_total,
            "fp": fp_total,
            "fn": fn_total,
            "precision": round(micro_p, 4),
            "recall": round(micro_r, 4),
            "f1": round(micro_f1, 4),
        }
    ]
    for light in sorted(by_lighting):
        tp, fp, fn = by_lighting[light]
        p, r, f1 = prf(tp, fp, fn)
        summary_rows.append(
            {
                "lighting": light,
                "n_gt": None,
                "n_pred": None,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f1, 4),
            }
        )

    written = write_excel(
        report_path,
        {
            "summary": pd.DataFrame(summary_rows),
            "per_pair": pd.DataFrame(per_pair_rows),
        },
    )

    print(f"GT matches: {gt_matches_path} ({len(gt_df)} rows, {len(gt_by_pair)} pairs)")
    print(f"Pairs evaluated: {len(pair_keys)} ({len(pred_by_pair)} with predictions)")
    print(f"Micro P/R/F1: {micro_p:.4f} / {micro_r:.4f} / {micro_f1:.4f}  (TP={tp_total} FP={fp_total} FN={fn_total})")
    print(f"Wrote {written}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Object matching precision/recall/F1 vs objects_matches.xlsx."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--gt-matches", type=Path, default=GT_MATCHES_XLSX)
    parser.add_argument("--output", type=Path, default=OBJECT_MATCHING_REPORT_XLSX)
    args = parser.parse_args()
    evaluate_object_matching(
        manifest_path=args.manifest,
        gt_matches_path=args.gt_matches,
        report_path=args.output,
    )


if __name__ == "__main__":
    main()
