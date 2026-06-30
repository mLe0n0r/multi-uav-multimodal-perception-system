#!/usr/bin/env python3
"""
Run detection and localization evaluation (separate Excel reports).

Ground truth (detection):  evaluation/groundTruth_data/visual/gt_labels/
Ground truth (localization): evaluation/groundTruth_data/visual/objects_position.xlsx
Predictions:   output/.cache/vision/{frame}.json or manifest visual.json paths
Outputs (visual):
  evaluation/results/visual_perception_eval/*.xlsx

Requires system vision outputs (from scripts/run_all_combinations.py --steps vision).
Triangulation eval needs multi-UAV runs (two views) in the manifest.

Usage (from repo root):
  python evaluation/scripts/evaluate_vision.py
  python evaluation/scripts/evaluate_detection.py
  python evaluation/scripts/evaluate_localization.py
  python evaluation/scripts/evaluate_triangulation_localization.py
  python evaluation/scripts/evaluate_object_matching.py
  python evaluation/scripts/evaluate_multi_uav_counts.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from eval_common import (  # noqa: E402
    DEFAULT_MANIFEST,
    DETECTION_REPORT_XLSX,
    LOCALIZATION_REPORT_XLSX,
    MULTI_UAV_COUNT_REPORT_XLSX,
    TRIANGULATION_LOCALIZATION_REPORT_XLSX,
)
from evaluate_detection import evaluate_detection  # noqa: E402
from evaluate_localization import evaluate_localization  # noqa: E402
from evaluate_multi_uav_counts import evaluate_multi_uav_counts  # noqa: E402
from evaluate_object_matching import evaluate_object_matching  # noqa: E402
from evaluate_triangulation_localization import (  # noqa: E402
    evaluate_triangulation_localization,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate system vision outputs (detection + localization, read-only)."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="evaluation_manifest.csv (fallback paths to visual.json)",
    )
    parser.add_argument(
        "--detection-output",
        type=Path,
        default=DETECTION_REPORT_XLSX,
        help="Detection Excel report",
    )
    parser.add_argument(
        "--localization-output",
        type=Path,
        default=LOCALIZATION_REPORT_XLSX,
        help="Localization Excel report",
    )
    parser.add_argument(
        "--detection-only",
        action="store_true",
        help="Skip localization report",
    )
    parser.add_argument(
        "--localization-only",
        action="store_true",
        help="Skip detection report",
    )
    parser.add_argument(
        "--skip-triangulation",
        action="store_true",
        help="Skip triangulation localization report",
    )
    parser.add_argument(
        "--skip-multi-uav-counts",
        action="store_true",
        help="Skip multi-UAV count error report",
    )
    args = parser.parse_args()

    only_flags = sum(
        1 for f in (args.detection_only, args.localization_only) if f
    )
    if only_flags > 1:
        parser.error("Use at most one of --detection-only and --localization-only")

    if not args.localization_only:
        print("=== Detection ===")
        evaluate_detection(manifest_path=args.manifest, report_path=args.detection_output)

    if not args.detection_only:
        print("\n=== Localization (mono, visual.json) ===")
        evaluate_localization(
            manifest_path=args.manifest, report_path=args.localization_output
        )

    if not args.detection_only and not args.localization_only and not args.skip_triangulation:
        print("\n=== Object matching (cross-view vs objects_matches.xlsx) ===")
        evaluate_object_matching(manifest_path=args.manifest)

        print("\n=== Localization (triangulation, GT-confirmed matches) ===")
        evaluate_triangulation_localization(manifest_path=args.manifest)

    if not args.detection_only and not args.localization_only and not args.skip_multi_uav_counts:
        print("\n=== Multi-UAV count error (cross_view) ===")
        evaluate_multi_uav_counts(
            manifest_path=args.manifest, report_path=MULTI_UAV_COUNT_REPORT_XLSX
        )


if __name__ == "__main__":
    main()
