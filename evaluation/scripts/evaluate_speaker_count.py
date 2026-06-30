#!/usr/bin/env python3
"""
Speaker-count evaluation for radio audio (diarization).

Ground truth: evaluation/groundTruth_data/audio/speaker_count.xlsx
Prediction:    analytics.speaker_count in perception/transcript.json
               (unique speakers in segments after transcribe_run.py)

One evaluation row per audio file (scenarioN_audioM), not per image run.

Output: evaluation/results/audio_perception_eval/speaker_count_metrics.xlsx

Usage (from repo root):
  python evaluation/scripts/evaluate_speaker_count.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from eval_common import (  # noqa: E402
    DEFAULT_MANIFEST,
    REPO_ROOT,
    SPEAKER_COUNT_REPORT_XLSX,
    SPEAKER_COUNT_XLSX,
    iter_audio_transcripts_from_manifest,
    load_speaker_count_gt,
    predicted_speaker_count_from_transcript,
    write_excel,
)


def _audio_stem_from_gt_name(audio_file: str) -> str:
    return Path(str(audio_file).strip()).stem


def evaluate_speaker_count(
    *,
    manifest_path: Path,
    gt_path: Path,
    report_path: Path,
) -> Path:
    gt_df = load_speaker_count_gt(gt_path)
    gt_by_stem: Dict[str, int] = {}
    for _, row in gt_df.iterrows():
        stem = _audio_stem_from_gt_name(row["audio"])
        gt_by_stem[stem] = int(row["speaker_count"])

    pred_by_stem: Dict[str, dict] = {}
    for entry in iter_audio_transcripts_from_manifest(manifest_path):
        stem = entry["audio_stem"]
        count = predicted_speaker_count_from_transcript(entry["transcript_path"])
        pred_by_stem[stem] = {**entry, "pred_speaker_count": count}

    per_audio_rows: List[dict] = []
    errors: List[float] = []
    correct = 0
    evaluated = 0
    missing_pred = 0

    for stem in sorted(gt_by_stem):
        gt_n = gt_by_stem[stem]
        entry = pred_by_stem.get(stem, {})
        pred_n: Optional[int] = entry.get("pred_speaker_count")
        m = re.match(r"^(scenario\d+)_audio(\d+)$", stem)
        scenario = m.group(1) if m else entry.get("scenario", "")
        audio_idx = int(m.group(2)) if m else None

        if pred_n is None:
            missing_pred += 1
            row = {
                "audio": f"{stem}.wav",
                "scenario": scenario,
                "audio_index": audio_idx,
                "gt_speaker_count": gt_n,
                "pred_speaker_count": pd.NA,
                "error": pd.NA,
                "correct": False,
                "transcript_path": "",
                "run_id": "",
            }
        else:
            evaluated += 1
            err = abs(int(pred_n) - gt_n)
            is_correct = int(pred_n) == gt_n
            if is_correct:
                correct += 1
            errors.append(float(err))
            rel = ""
            if entry.get("transcript_path"):
                try:
                    rel = str(entry["transcript_path"].relative_to(REPO_ROOT))
                except ValueError:
                    rel = str(entry["transcript_path"])
            row = {
                "audio": f"{stem}.wav",
                "scenario": scenario,
                "audio_index": audio_idx,
                "gt_speaker_count": gt_n,
                "pred_speaker_count": int(pred_n),
                "error": err,
                "correct": is_correct,
                "transcript_path": rel,
                "run_id": entry.get("run_id", ""),
            }
        per_audio_rows.append(row)

    summary_rows: List[dict] = [
        {
            "n_audio": len(gt_by_stem),
            "n_evaluated": evaluated,
            "n_missing_pred": missing_pred,
            "n_correct": correct,
            "accuracy": round(correct / evaluated, 4) if evaluated else None,
            "mae": round(float(np.mean(errors)), 4) if errors else None,
        }
    ]

    written = write_excel(
        report_path,
        {
            "summary": pd.DataFrame(summary_rows),
            "per_audio": pd.DataFrame(per_audio_rows),
        },
    )

    print(f"GT: {gt_path} ({len(gt_by_stem)} audio files)")
    print(f"Evaluated: {evaluated}/{len(gt_by_stem)}  missing transcript: {missing_pred}")
    if evaluated:
        print(f"Accuracy (exact count): {correct}/{evaluated} = {correct/evaluated:.2%}")
        print(f"MAE (abs(pred - GT)): {np.mean(errors):.4f}")
    print(f"Wrote {written}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate diarization speaker_count vs speaker_count.xlsx."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--gt", type=Path, default=SPEAKER_COUNT_XLSX)
    parser.add_argument("--output", type=Path, default=SPEAKER_COUNT_REPORT_XLSX)
    args = parser.parse_args()
    evaluate_speaker_count(
        manifest_path=args.manifest,
        gt_path=args.gt,
        report_path=args.output,
    )


if __name__ == "__main__":
    main()
