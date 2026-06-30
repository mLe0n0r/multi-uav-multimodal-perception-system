#!/usr/bin/env python3
"""
Scene object-count accuracy vs objects_count.xlsx.

Ground truth (mono and multi): objects_count.xlsx sheet 'per_scenario'
  (deduplicated scene totals per scenario).

Mono-UAV (imgN_audM):
  - vision-only:      visual.json
  - vision and audio: visual/cross_view baseline + audio_only (radio gap only
    for entities not already detected; on-scene speakers are roles, not extras)

Multi-UAV:
  - vision-only:      cross_view counts_after_match
  - vision and audio: counts_after_match + audio_only from sls.json
    (equals counts_by_class when SLS is reconciled; never below vision-only)

Output: evaluation/results/sls_eval/scene_count_metrics.xlsx

Usage (from repo root):
  python evaluation/scripts/evaluate_scene_counts.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from eval_common import (  # noqa: E402
    COUNT_CLASS_NAMES,
    DEFAULT_MANIFEST,
    GT_OBJECTS_COUNT_XLSX,
    GT_SCENARIO_COUNT_SHEET,
    REPO_ROOT,
    SCENE_COUNT_REPORT_XLSX,
    build_img_metadata,
    counts_from_sls,
    counts_vision_and_audio,
    counts_from_visual_json,
    cross_view_result_for_run,
    fused_counts_from_cross_view,
    iter_mono_uav_runs,
    iter_multi_uav_runs,
    load_class_names,
    load_scenario_count_gt,
    MEAN_RELATIVE_COUNT_ERROR_COL,
    mean_relative_count_error,
    relative_count_error,
    sls_audio_only_deltas,
    sls_reconciles_with_baseline,
    write_excel,
)

SOURCE_VISION_ONLY = "vision-only"
SOURCE_VISION_AND_AUDIO = "vision and audio"
CONFIG_MONO = "mono"
CONFIG_MULTI = "multi"


def _append_rows(
    rows: List[dict],
    *,
    config: str,
    source: str,
    run_id: str,
    scenario_num: int,
    img_ref: int,
    lighting: str,
    gt_ref: str,
    pred_counts: Dict[str, int],
    gt_counts: Dict[str, int],
    pred_origin: str,
    audio_deltas: Dict[str, int] | None = None,
    visual_baseline: Dict[str, int] | None = None,
    sls_reconciles: bool | None = None,
    sls_raw_counts: Dict[str, int] | None = None,
) -> None:
    for cls_name in COUNT_CLASS_NAMES:
        gt_c = int(gt_counts.get(cls_name, 0))
        pred_c = int(pred_counts.get(cls_name, 0))
        row = {
            "config": config,
            "source": source,
            "scenario": f"scenario{scenario_num}",
            "run_id": run_id,
            "img_ref": f"img{img_ref}",
            "gt_ref": gt_ref,
            "lighting": lighting,
            "class": cls_name,
            "gt_count": gt_c,
            "pred_count": pred_c,
            "count_error": pred_c - gt_c,
            "relative_count_error": relative_count_error(pred_c, gt_c),
            "pred_origin": pred_origin,
        }
        if audio_deltas is not None:
            row["audio_only_delta"] = int(audio_deltas.get(cls_name, 0))
        if visual_baseline is not None:
            row["visual_baseline"] = int(visual_baseline.get(cls_name, 0))
        if sls_reconciles is not None:
            row["sls_reconciles"] = sls_reconciles
        if sls_raw_counts is not None:
            row["sls_counts_by_class"] = int(sls_raw_counts.get(cls_name, 0))
        rows.append(row)


def _summarize(rows: List[dict]) -> pd.DataFrame:
    summary: List[dict] = []
    keys = sorted({(r["config"], r["source"], r["class"]) for r in rows})
    for config, source, cls_name in keys:
        subset = [
            r
            for r in rows
            if r["config"] == config
            and r["source"] == source
            and r["class"] == cls_name
        ]
        if not subset:
            continue
        summary.append(
            {
                "config": config,
                "source": source,
                "class": cls_name,
                "n_evaluated": len(subset),
                MEAN_RELATIVE_COUNT_ERROR_COL: mean_relative_count_error(subset),
            }
        )
    return pd.DataFrame(summary)


def evaluate_scene_counts(*, manifest_path: Path, report_path: Path) -> Path:
    class_names = load_class_names()
    img_lighting = build_img_metadata()
    gt_by_scenario = load_scenario_count_gt()
    per_run_rows: List[dict] = []

    n_mono = n_multi = 0
    mono_vision_seen: set[int] = set()
    for run in iter_mono_uav_runs(manifest_path):
        n_mono += 1
        gt_scene = gt_by_scenario.get(run.scenario_num)
        if not gt_scene:
            continue
        lighting = img_lighting.get(run.img_ref, "unknown")
        visual_payload = json.loads(run.visual_json.read_text(encoding="utf-8"))
        pred_visual = counts_from_visual_json(visual_payload)
        if run.img_ref not in mono_vision_seen:
            mono_vision_seen.add(run.img_ref)
            _append_rows(
                per_run_rows,
                config=CONFIG_MONO,
                source=SOURCE_VISION_ONLY,
                run_id=run.run_id,
                scenario_num=run.scenario_num,
                img_ref=run.img_ref,
                lighting=lighting,
                gt_ref=f"scenario{run.scenario_num}",
                pred_counts=pred_visual,
                gt_counts=gt_scene,
                pred_origin=str(run.visual_json.relative_to(REPO_ROOT)),
            )
        if run.sls_json is not None:
            sls_payload = json.loads(run.sls_json.read_text(encoding="utf-8"))
            pred_va = counts_vision_and_audio(pred_visual, sls_payload)
            _append_rows(
                per_run_rows,
                config=CONFIG_MONO,
                source=SOURCE_VISION_AND_AUDIO,
                run_id=run.run_id,
                scenario_num=run.scenario_num,
                img_ref=run.img_ref,
                lighting=lighting,
                gt_ref=f"scenario{run.scenario_num}",
                pred_counts=pred_va,
                gt_counts=gt_scene,
                pred_origin=str(run.sls_json.relative_to(REPO_ROOT)),
                audio_deltas=sls_audio_only_deltas(sls_payload),
                visual_baseline=pred_visual,
                sls_reconciles=sls_reconciles_with_baseline(pred_visual, sls_payload),
                sls_raw_counts=counts_from_sls(sls_payload),
            )

    for run in iter_multi_uav_runs(manifest_path):
        n_multi += 1
        if run.scenario_num not in gt_by_scenario:
            continue
        gt_scene = gt_by_scenario[run.scenario_num]
        lighting = img_lighting.get(run.img_ref, "unknown")
        cv_payload = cross_view_result_for_run(run)
        pred_cv = fused_counts_from_cross_view(cv_payload)
        if pred_cv is not None:
            cv_origin = (
                str(run.cross_view_json.relative_to(REPO_ROOT))
                if run.cross_view_json
                else "(computed)"
            )
            _append_rows(
                per_run_rows,
                config=CONFIG_MULTI,
                source=SOURCE_VISION_ONLY,
                run_id=run.run_id,
                scenario_num=run.scenario_num,
                img_ref=run.img_ref,
                lighting=lighting,
                gt_ref=f"scenario{run.scenario_num}",
                pred_counts=pred_cv,
                gt_counts=gt_scene,
                pred_origin=cv_origin,
            )
        if run.sls_json is not None and pred_cv is not None:
            sls_payload = json.loads(run.sls_json.read_text(encoding="utf-8"))
            pred_va = counts_vision_and_audio(pred_cv, sls_payload)
            _append_rows(
                per_run_rows,
                config=CONFIG_MULTI,
                source=SOURCE_VISION_AND_AUDIO,
                run_id=run.run_id,
                scenario_num=run.scenario_num,
                img_ref=run.img_ref,
                lighting=lighting,
                gt_ref=f"scenario{run.scenario_num}",
                pred_counts=pred_va,
                gt_counts=gt_scene,
                pred_origin=str(run.sls_json.relative_to(REPO_ROOT)),
                audio_deltas=sls_audio_only_deltas(sls_payload),
                visual_baseline=pred_cv,
                sls_reconciles=sls_reconciles_with_baseline(pred_cv, sls_payload),
                sls_raw_counts=counts_from_sls(sls_payload),
            )

    summary_df = _summarize(per_run_rows)
    per_run_df = pd.DataFrame(per_run_rows)
    if not per_run_df.empty:
        per_run_df["class_label"] = per_run_df["class"].map(
            {name: class_names.get(i, name) for i, name in enumerate(COUNT_CLASS_NAMES)}
        )

    written = write_excel(
        report_path,
        {
            "metrics": summary_df,
            "per_run": per_run_df,
        },
    )

    print(f"Manifest runs: mono={n_mono} multi={n_multi}")
    print(
        f"GT counts: {GT_OBJECTS_COUNT_XLSX} sheet '{GT_SCENARIO_COUNT_SHEET}' (per scenario)"
    )
    print(f"Wrote {written}")
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scene count accuracy: visual/cross_view/sls vs objects_count.xlsx."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=SCENE_COUNT_REPORT_XLSX)
    args = parser.parse_args()
    evaluate_scene_counts(manifest_path=args.manifest, report_path=args.output)


if __name__ == "__main__":
    main()
