"""
Standard run directory layout for perception + fusion experiments.

  output/<scenario>/<lighting_condition>/<run_id>/
    perception/
      visual.json              # single-image run
      visual_<view_id>.json    # multi-image run (flat, no subfolders)
      transcript.json
    fusion/
      cross_view.json          # multi-UAV: same_incident + visuals + matches
      llm_output.json
      sls.json

Run id examples: img0_aud1, img0_img12_aud1

Lighting folders (time-of-day group):
  tarde -> afternoon_light
  noite -> night_light
  dia   -> daylight
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PERCEPTION_DIR = "perception"
FUSION_DIR = "fusion"

VISUAL_MONO_NAME = "visual.json"
VISUAL_MULTI_PATTERN = re.compile(r"^visual_(.+)\.json$")
TRANSCRIPT_NAME = "transcript.json"
CROSS_VIEW_NAME = "cross_view.json"
LLM_OUTPUT_NAME = "llm_output.json"
SLS_NAME = "sls.json"

# Legacy paths (read-only fallback)
_LEGACY_CROSS_VIEW_PATHS = (
    "fusion/matching/geometric.json",
    "fusion/matching/result.json",
)


def build_run_id(image_ids: List[str], audio_id: str) -> str:
    """e.g. ['img0','img12'], 'aud1' -> img0_img12_aud1"""
    if not image_ids:
        raise ValueError("At least one image id required")
    if not audio_id:
        raise ValueError("audio_id required")
    return "_".join(image_ids + [audio_id])


GROUP_TO_LIGHTING_FOLDER = {
    "tarde": "afternoon_light",
    "noite": "night_light",
    "dia": "daylight",
}


def lighting_folder_for_group(group: str) -> str:
    """Map batch group name to output lighting folder."""
    return GROUP_TO_LIGHTING_FOLDER.get(group, group)


def run_dir_path(
    scenario: str,
    run_id: str,
    *,
    group: Optional[str] = None,
    lighting_condition: Optional[str] = None,
    output_base: Path | str = "output",
) -> Path:
    """Canonical run directory: output/<scenario>/<lighting>/<run_id>/."""
    lighting = lighting_condition or (
        lighting_folder_for_group(group) if group else None
    )
    if not lighting:
        raise ValueError("Provide group or lighting_condition")
    return Path(output_base) / scenario / lighting / run_id


def run_root(scenario: str, run_id: str, output_base: Path | str = "output") -> Path:
    """Deprecated layout without lighting folder; prefer run_dir_path()."""
    return Path(output_base) / scenario / run_id


def perception_dir(run_dir: Path | str) -> Path:
    return Path(run_dir) / PERCEPTION_DIR


def fusion_dir(run_dir: Path | str) -> Path:
    return Path(run_dir) / FUSION_DIR


def cross_view_path(run_dir: Path | str) -> Path:
    return fusion_dir(run_dir) / CROSS_VIEW_NAME


def _legacy_cross_view_paths(run_dir: Path | str) -> List[Path]:
    root = Path(run_dir)
    return [root / rel for rel in _LEGACY_CROSS_VIEW_PATHS]


def load_cross_view(run_dir: Path | str) -> Optional[Dict[str, Any]]:
    """Load fusion/cross_view.json, or legacy matching/*.json if present."""
    for path in [cross_view_path(run_dir), *_legacy_cross_view_paths(run_dir)]:
        if path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


def visual_views_from_cross_view(bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    views: List[Dict[str, Any]] = []
    for view_id in bundle.get("views", []):
        data = copy.deepcopy((bundle.get("visuals") or {}).get(view_id))
        if data is None:
            continue
        data["_view_id"] = view_id
        views.append(data)
    return views


def visual_mono_path(run_dir: Path | str) -> Path:
    return perception_dir(run_dir) / VISUAL_MONO_NAME


def visual_multi_path(run_dir: Path | str, view_id: str) -> Path:
    return perception_dir(run_dir) / f"visual_{view_id}.json"


def transcript_path(run_dir: Path | str) -> Path:
    return perception_dir(run_dir) / TRANSCRIPT_NAME


def llm_output_path(run_dir: Path | str) -> Path:
    return fusion_dir(run_dir) / LLM_OUTPUT_NAME


def sls_path(run_dir: Path | str) -> Path:
    return fusion_dir(run_dir) / SLS_NAME


def sls_path_for_view(run_dir: Path | str, view_id: str) -> Path:
    """Per-view SLS when multi-UAV views are independent incidents."""
    return fusion_dir(run_dir) / f"sls_{view_id}.json"


def is_multi_view_run(run_dir: Path | str) -> bool:
    return len(discover_visual_views(run_dir)) > 1


def discover_visual_views(run_dir: Path | str) -> List[Tuple[str, Path]]:
    """
    Return [(view_id, path), ...] sorted by view_id.
    Mono run: one entry ('mono', perception/visual.json) if present.
    Multi: perception/visual_<view_id>.json
    """
    pdir = perception_dir(run_dir)
    if not pdir.is_dir():
        return []

    multi: List[Tuple[str, Path]] = []
    for path in sorted(pdir.glob("visual_*.json")):
        match = VISUAL_MULTI_PATTERN.match(path.name)
        if match:
            multi.append((match.group(1), path))

    if multi:
        return multi

    mono = pdir / VISUAL_MONO_NAME
    if mono.is_file():
        return [("mono", mono)]

    return []


def load_visual_views(run_dir: Path | str) -> List[Dict[str, Any]]:
    views: List[Dict[str, Any]] = []
    for view_id, path in discover_visual_views(run_dir):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("_view_id", view_id)
        views.append(data)
    return views


# Backwards-compatible aliases
def load_geometric_result(run_dir: Path | str) -> Optional[Dict[str, Any]]:
    return load_cross_view(run_dir)


def load_matching_result(run_dir: Path | str) -> Optional[Dict[str, Any]]:
    return load_cross_view(run_dir)


def geometric_result_path(run_dir: Path | str) -> Path:
    return cross_view_path(run_dir)


def visual_views_from_geometric(bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    return visual_views_from_cross_view(bundle)


def ensure_run_dirs(run_dir: Path | str) -> None:
    perception_dir(run_dir).mkdir(parents=True, exist_ok=True)
    fusion_dir(run_dir).mkdir(parents=True, exist_ok=True)


def primary_visual_path(run_dir: Path | str) -> Path:
    """First view path for SLS object geometry (legacy single-visual runs)."""
    views = discover_visual_views(run_dir)
    if not views:
        raise FileNotFoundError(f"No visual JSON in {perception_dir(run_dir)}")
    return views[0][1]


def run_has_complete_sls(run_dir: Path | str) -> bool:
    """Mono or fused multi: fusion/sls.json; independent multi: sls_<view_id>.json each."""
    if not is_multi_view_run(run_dir):
        return sls_path(run_dir).is_file()
    matching = load_cross_view(run_dir)
    if matching and matching.get("same_incident"):
        return sls_path(run_dir).is_file()
    views = discover_visual_views(run_dir)
    if len(views) > 1:
        return all(sls_path_for_view(run_dir, vid).is_file() for vid, _ in views)
    return sls_path(run_dir).is_file()


def load_primary_visual(run_dir: Path | str) -> Dict[str, Any]:
    with open(primary_visual_path(run_dir), "r", encoding="utf-8") as f:
        return json.load(f)
