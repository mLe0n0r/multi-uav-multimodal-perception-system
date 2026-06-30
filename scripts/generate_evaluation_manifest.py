#!/usr/bin/env python3
"""Generate evaluation_manifest.csv for offline perception/fusion evaluation."""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fusion.run_layout import (
    cross_view_path,
    lighting_folder_for_group,
    run_dir_path,
    run_has_complete_sls,
    transcript_path,
    visual_mono_path,
    visual_multi_path,
)
from scripts.run_all_combinations import (
    IMG_ROOT,
    POSE_DIR,
    SCENARIOS,
    img_view_id,
    iter_plans,
    resolve_assets,
)

VISION_RESULTS = _REPO_ROOT / "perception" / "vision" / "results"
DETECTION_LABEL_DIR = VISION_RESULTS / "labels"

def rel(path: Path) -> str:
    return str(path.relative_to(_REPO_ROOT)).replace("\\", "/")


def main() -> None:
    rows: list[dict] = []
    for plan in iter_plans(SCENARIOS):
        lighting = lighting_folder_for_group(plan.group)
        run_dir = run_dir_path(
            plan.scenario,
            plan.run_id,
            group=plan.group,
            output_base=_REPO_ROOT / "output",
        )

        aud_match = re.search(r"_audio(\d+)$", plan.audio_stem)
        row: dict = {
            "scenario": plan.scenario,
            "group": plan.group,
            "lighting_condition": lighting,
            "scenario_folder": plan.scenario_folder,
            "vision_mode": plan.mode,
            "run_id": plan.run_id,
            "multi_uav": plan.multi,
            "n_uavs": len(plan.image_nums),
            "image_nums": ",".join(str(n) for n in plan.image_nums),
            "view_ids": ",".join(img_view_id(n) for n in plan.image_nums),
            "audio_stem": plan.audio_stem,
            "aud_id": f"aud{aud_match.group(1)}" if aud_match else "",
            "run_dir": rel(run_dir),
            "has_output_dir": run_dir.is_dir(),
            "has_sls": run_has_complete_sls(run_dir),
            "transcript_path": rel(transcript_path(run_dir))
            if transcript_path(run_dir).is_file()
            else "",
            "cross_view_path": rel(cross_view_path(run_dir))
            if cross_view_path(run_dir).is_file()
            else "",
        }

        for idx, n in enumerate(plan.image_nums[:2]):
            vid = img_view_id(n)
            assets = resolve_assets(n, img_root=IMG_ROOT, pose_dir=POSE_DIR)
            frame_id = f"{n:05d}"
            det_label = DETECTION_LABEL_DIR / f"{frame_id}.txt"
            prefix = f"view{idx + 1}_"
            row[prefix + "view_id"] = vid
            row[prefix + "image_num"] = n
            row[prefix + "frame_id"] = frame_id
            if assets:
                row[prefix + "image_path"] = rel(assets["img"])
                row[prefix + "telemetry_path"] = rel(assets["telemetry"])
                row[prefix + "vision_inputs_ok"] = True
            else:
                row[prefix + "vision_inputs_ok"] = False
            row[prefix + "label_path"] = rel(det_label) if det_label.is_file() else ""

            if plan.multi:
                vp = visual_multi_path(run_dir, vid)
            else:
                vp = visual_mono_path(run_dir)
            row[prefix + "visual_pred_path"] = rel(vp) if vp.is_file() else ""

        rows.append(row)

    base_cols = [
        "scenario",
        "group",
        "lighting_condition",
        "scenario_folder",
        "vision_mode",
        "run_id",
        "multi_uav",
        "n_uavs",
        "image_nums",
        "view_ids",
        "audio_stem",
        "aud_id",
        "run_dir",
        "has_output_dir",
        "has_sls",
        "transcript_path",
        "cross_view_path",
    ]
    view_cols = []
    for i in (1, 2):
        for c in (
            "view_id",
            "image_num",
            "frame_id",
            "image_path",
            "label_path",
            "telemetry_path",
            "visual_pred_path",
            "vision_inputs_ok",
        ):
            view_cols.append(f"view{i}_{c}")

    out = _REPO_ROOT / "evaluation" / "evaluation_manifest.csv"
    fieldnames = base_cols + view_cols
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    has_sls = sum(1 for r in rows if r.get("has_sls"))
    has_out = sum(1 for r in rows if r.get("has_output_dir"))
    print(f"Wrote {len(rows)} rows to {out}")
    print(f"has_output_dir={has_out} has_sls={has_sls}")


if __name__ == "__main__":
    main()
