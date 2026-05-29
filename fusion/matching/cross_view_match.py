"""
Cross-view multi-UAV matching (objectMatching.ipynb).

1. Check whether views cover the same incident (ground footprint overlap).
2. If yes, match objects via match_frames_from_loaded / match_two_frames.
3. Write fusion/cross_view.json: original visuals + cross-view matches.

Usage:
  python fusion/matching/cross_view_match.py --run-dir output/scenario1/img0_img01_aud1

With raw assets (same paths as the notebook):
  python fusion/matching/cross_view_match.py --run-dir ... \\
    --scenario-folder scenario1D --img-root perception/vision/data/results/annotated_imgs
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_MATCHING_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _MATCHING_DIR.parents[1]
_VISION_DATA = _REPO_ROOT / "perception" / "vision" / "data"
_FUSION_ROOT = _MATCHING_DIR.parent
for path in (_FUSION_ROOT, _MATCHING_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from object_matching_core import (
    FOV_X_DEG,
    count_by_object_type_after_matching,
    load_frame,
    load_frame_from_visual,
    match_frames_from_loaded,
)
from run_layout import cross_view_path, discover_visual_views, ensure_run_dirs

SAME_INCIDENT_OVERLAP_THRESHOLD = 0.15


def load_json(path: Path | str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_json_default)


def parse_metric(value: Any) -> float:
    if value is None:
        return 0.0
    text = str(value).strip().lower()
    match = re.match(r"([\d.+-]+)\s*(m|deg|°)?", text)
    if match:
        return float(match.group(1))
    return float(text)


def format_position_meters(x: float, y: float, z: float) -> Dict[str, str]:
    return {
        "x": f"{x:.2f} m",
        "y": f"{y:.2f} m",
        "z": f"{z:.2f} m",
    }


def image_size_from_visual(visual: Dict[str, Any], default_w: int = 1280, default_h: int = 720) -> Tuple[int, int]:
    size = visual.get("image_size") or {}
    return int(size.get("width") or default_w), int(size.get("height") or default_h)


def camera_xy(visual: Dict[str, Any]) -> Tuple[float, float]:
    pos = (visual.get("camera") or {}).get("position") or {}
    return parse_metric(pos.get("x")), parse_metric(pos.get("y"))


def camera_altitude_m(visual: Dict[str, Any]) -> float:
    pos = (visual.get("camera") or {}).get("position") or {}
    return abs(parse_metric(pos.get("z")))


def ground_coverage_radius_m(visual: Dict[str, Any], fov_x_deg: float = FOV_X_DEG) -> float:
    z = camera_altitude_m(visual)
    return z * math.tan(math.radians(fov_x_deg / 2.0))


def _circle_area(r: float) -> float:
    return math.pi * r * r


def _circle_intersection_area(r1: float, r2: float, d: float) -> float:
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return _circle_area(min(r1, r2))
    part1 = r1 * r1 * math.acos((d * d + r1 * r1 - r2 * r2) / (2 * d * r1))
    part2 = r2 * r2 * math.acos((d * d + r2 * r2 - r1 * r1) / (2 * d * r2))
    part3 = 0.5 * math.sqrt(
        max(0.0, (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
    )
    return part1 + part2 - part3


def footprint_overlap_ratio(
    visual_a: Dict[str, Any],
    visual_b: Dict[str, Any],
    fov_x_deg: float = FOV_X_DEG,
) -> float:
    r1 = ground_coverage_radius_m(visual_a, fov_x_deg)
    r2 = ground_coverage_radius_m(visual_b, fov_x_deg)
    x1, y1 = camera_xy(visual_a)
    x2, y2 = camera_xy(visual_b)
    d = math.hypot(x2 - x1, y2 - y1)
    inter = _circle_intersection_area(r1, r2, d)
    union = _circle_area(r1) + _circle_area(r2) - inter
    if union <= 1e-9:
        return 0.0
    return float(inter / union)


def check_same_incident(
    views: List[Tuple[str, Dict[str, Any]]],
    *,
    fov_x_deg: float = FOV_X_DEG,
    threshold: float = SAME_INCIDENT_OVERLAP_THRESHOLD,
) -> Tuple[bool, float]:
    if len(views) < 2:
        return False, 0.0
    ratios: List[float] = []
    for i in range(len(views)):
        for j in range(i + 1, len(views)):
            ratios.append(footprint_overlap_ratio(views[i][1], views[j][1], fov_x_deg=fov_x_deg))
    min_ratio = min(ratios)
    return min_ratio >= threshold, min_ratio


def view_id_to_frame_id(view_id: str) -> str:
    """e.g. img0 -> 00000, img01 -> 00001, img12 -> 00012 (notebook frame filenames)."""
    match = re.match(r"^img(\d+)$", view_id, re.IGNORECASE)
    if match:
        return f"{int(match.group(1)):05d}"
    if view_id.isdigit():
        return f"{int(view_id):05d}"
    return view_id


def resolve_notebook_assets(
    view_id: str,
    *,
    img_root: Path,
    label_dir: Path,
    pose_dir: Path,
    scenario_folder: Optional[str] = None,
) -> Optional[Dict[str, Path]]:
    """Resolve img/label/telemetry paths like objectMatching.ipynb usage cell."""
    frame_id = view_id_to_frame_id(view_id)
    folder = scenario_folder or ""

    img_candidates = [
        img_root / folder / f"{frame_id}.png",
        img_root / f"{frame_id}.png",
    ]
    label_candidates = [
        label_dir / folder / f"{frame_id}.txt",
        label_dir / f"{frame_id}.txt",
    ]
    telemetry_candidates = [
        pose_dir / folder / f"{frame_id}.txt",
        pose_dir / f"{frame_id}.txt",
    ]

    img_path = next((p for p in img_candidates if p.is_file()), None)
    label_path = next((p for p in label_candidates if p.is_file()), None)
    telemetry_path = next((p for p in telemetry_candidates if p.is_file()), None)

    if img_path and label_path and telemetry_path:
        return {
            "img": img_path,
            "label": label_path,
            "telemetry": telemetry_path,
        }
    return None


def _tri_to_position(tri: Any) -> Optional[Dict[str, str]]:
    if tri is None:
        return None
    if hasattr(tri, "__len__") and len(tri) >= 3:
        return format_position_meters(float(tri[0]), float(tri[1]), float(tri[2]))
    return None


def _raw_matches_to_cross_view(
    raw: List[Dict[str, Any]],
    frame_ref: Dict[str, Any],
    frame_other: Dict[str, Any],
    view_ref: str,
    view_other: str,
) -> List[Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []
    for m in raw:
        i, j = int(m["A"]), int(m["B"])
        tri = m.get("tri_P_w")
        pos = _tri_to_position(tri) if m.get("triangulation_ok") and tri is not None else None
        matches.append(
            {
                "view_a": view_ref,
                "id_a": int(frame_ref["object_ids"][i]),
                "view_b": view_other,
                "id_b": int(frame_other["object_ids"][j]),
                "position": pos,
            }
        )
    return matches


def _build_frame(
    visual: Dict[str, Any],
    assets: Optional[Dict[str, Path]],
    *,
    fov_x_deg: float,
) -> Dict[str, Any]:
    """Notebook-identical frame when all raw assets exist; else visual JSON (+ optional paths)."""
    if assets and all(assets.get(k) for k in ("img", "label", "telemetry")):
        frame = load_frame(
            assets["img"],
            assets["label"],
            assets["telemetry"],
            fov_x_deg=fov_x_deg,
        )
        from object_matching_core import labels_and_ids_from_visual

        _, visual_ids = labels_and_ids_from_visual(visual)
        if len(visual_ids) == len(frame["labels"]):
            frame["object_ids"] = visual_ids
        return frame

    return load_frame_from_visual(
        visual,
        img_path=str(assets["img"]) if assets and assets.get("img") else None,
        telemetry_path=str(assets["telemetry"]) if assets and assets.get("telemetry") else None,
        fov_x_deg=fov_x_deg,
    )


def _match_pair(
    visual_ref: Dict[str, Any],
    visual_other: Dict[str, Any],
    view_ref: str,
    view_other: str,
    *,
    fov_x_deg: float,
    assets_ref: Optional[Dict[str, Path]] = None,
    assets_other: Optional[Dict[str, Path]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    frame_ref = _build_frame(visual_ref, assets_ref, fov_x_deg=fov_x_deg)
    frame_other = _build_frame(visual_other, assets_other, fov_x_deg=fov_x_deg)
    raw, unmatched_a, unmatched_b, _, _, dets_a, dets_b = match_frames_from_loaded(
        frame_ref, frame_other
    )
    matches = _raw_matches_to_cross_view(raw, frame_ref, frame_other, view_ref, view_other)
    counts = count_by_object_type_after_matching(
        raw, unmatched_a, unmatched_b, dets_a, dets_b
    )
    return matches, counts


def run_cross_view_match(
    views: List[Tuple[str, Dict[str, Any]]],
    *,
    fov_x_deg: float = FOV_X_DEG,
    overlap_threshold: float = SAME_INCIDENT_OVERLAP_THRESHOLD,
    img_root: Optional[Path] = None,
    label_dir: Optional[Path] = None,
    pose_dir: Optional[Path] = None,
    scenario_folder: Optional[str] = None,
) -> Dict[str, Any]:
    if len(views) < 2:
        raise ValueError("Cross-view matching requires at least two visual.json views")

    view_ids = [v[0] for v in views]
    originals = {vid: copy.deepcopy(vis) for vid, vis in views}

    same_incident, _ = check_same_incident(
        views, fov_x_deg=fov_x_deg, threshold=overlap_threshold
    )

    matches: List[Dict[str, Any]] = []
    notebook_counts: Optional[Dict[str, int]] = None

    if same_incident:
        ref_id, ref_visual = views[0]
        assets_by_view: Dict[str, Optional[Dict[str, Path]]] = {}
        if img_root and label_dir and pose_dir:
            for vid, _ in views:
                assets_by_view[vid] = resolve_notebook_assets(
                    vid,
                    img_root=img_root,
                    label_dir=label_dir,
                    pose_dir=pose_dir,
                    scenario_folder=scenario_folder,
                )

        ref_assets = assets_by_view.get(ref_id) if img_root else None
        for other_id, other_visual in views[1:]:
            other_assets = assets_by_view.get(other_id) if img_root else None
            pair_matches, counts = _match_pair(
                ref_visual,
                other_visual,
                ref_id,
                other_id,
                fov_x_deg=fov_x_deg,
                assets_ref=ref_assets,
                assets_other=other_assets,
            )
            matches.extend(pair_matches)
            notebook_counts = counts

    result: Dict[str, Any] = {
        "same_incident": same_incident,
        "views": view_ids,
        "visuals": originals,
        "matches": matches,
    }
    if notebook_counts is not None:
        result["notebook_counts_after_match"] = notebook_counts
    return result


def run_cross_view_for_run_dir(
    run_dir: Path | str,
    *,
    fov_x_deg: float = FOV_X_DEG,
    overlap_threshold: float = SAME_INCIDENT_OVERLAP_THRESHOLD,
    img_root: Optional[Path] = None,
    label_dir: Optional[Path] = None,
    pose_dir: Optional[Path] = None,
    scenario_folder: Optional[str] = None,
) -> Dict[str, Any]:
    discovered = discover_visual_views(run_dir)
    if len(discovered) < 2:
        raise ValueError(
            f"Need at least two perception/visual_<id>.json under {run_dir}; found {len(discovered)}"
        )
    views = [(vid, load_json(path)) for vid, path in discovered]
    result = run_cross_view_match(
        views,
        fov_x_deg=fov_x_deg,
        overlap_threshold=overlap_threshold,
        img_root=img_root,
        label_dir=label_dir,
        pose_dir=pose_dir,
        scenario_folder=scenario_folder,
    )
    out_path = cross_view_path(run_dir)
    ensure_run_dirs(run_dir)
    save_json(result, out_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-view multi-UAV match → fusion/cross_view.json (objectMatching.ipynb)"
    )
    parser.add_argument("--run-dir", required=True, help="Run folder with perception/visual_<id>.json")
    parser.add_argument(
        "--img-root",
        type=Path,
        default=_VISION_DATA / "results" / "annotated_imgs",
        help="Notebook img_root_dir (PNG per frame)",
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=_VISION_DATA / "results" / "labels",
        help="Notebook label_dir (YOLO txt)",
    )
    parser.add_argument(
        "--pose-dir",
        type=Path,
        default=_VISION_DATA / "telemetryData",
        help="Notebook pose_dir (telemetry txt)",
    )
    parser.add_argument(
        "--scenario-folder",
        default=None,
        help="Subfolder under img/label/pose (e.g. scenario1D). Omit to use flat layout.",
    )
    parser.add_argument(
        "--no-raw-assets",
        action="store_true",
        help="Only use perception/visual_*.json (no img/label/telemetry files)",
    )
    parser.add_argument("--fov-x", type=float, default=FOV_X_DEG)
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=SAME_INCIDENT_OVERLAP_THRESHOLD,
        help="Min ground footprint IoU to declare same incident",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    ensure_run_dirs(run_dir)

    img_root = None if args.no_raw_assets else args.img_root
    label_dir = None if args.no_raw_assets else args.label_dir
    pose_dir = None if args.no_raw_assets else args.pose_dir

    result = run_cross_view_for_run_dir(
        run_dir,
        fov_x_deg=args.fov_x,
        overlap_threshold=args.overlap_threshold,
        img_root=img_root,
        label_dir=label_dir,
        pose_dir=pose_dir,
        scenario_folder=args.scenario_folder,
    )
    out_path = cross_view_path(run_dir)

    print(f"same_incident: {result['same_incident']}")
    print(f"views: {result['views']}")
    print(f"matches: {len(result['matches'])}")
    for m in result["matches"]:
        print(
            f"  {m['view_a']}:{m['id_a']} <-> {m['view_b']}:{m['id_b']}"
            + (f" @ {m['position']}" if m.get("position") else "")
        )
    if result.get("notebook_counts_after_match"):
        c = result["notebook_counts_after_match"]
        print(
            f"Counts pós-match (notebook): person={c.get('person')}, "
            f"vehicle={c.get('vehicle')}, emergency_vehicle={c.get('emergency_vehicle')}"
        )
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
