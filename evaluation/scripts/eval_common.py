from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import yaml

_EVAL_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = _EVAL_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_all_combinations import SCENARIOS, VISION_CACHE

GT_ROOT = _EVAL_DIR / "groundTruth_data"
GT_VISUAL_DIR = GT_ROOT / "visual"
GT_AUDIO_DIR = GT_ROOT / "audio"
GT_LABELS_DIR = GT_VISUAL_DIR / "gt_labels"
GT_POSITIONS_XLSX = GT_VISUAL_DIR / "objects_position.xlsx"
GT_OBJECTS_COUNT_CSV = GT_VISUAL_DIR / "objects_count.csv"
GT_OBJECTS_COUNT_XLSX = GT_VISUAL_DIR / "objects_count.xlsx"
GT_OBJECTS_COUNT_SHEET = "objects_count"
GT_SCENARIO_COUNT_SHEET = "per_scenario"
GT_ROLES_XLSX = GT_VISUAL_DIR / "gt_roles.xlsx"
GT_ROLES_PER_SCENARIO_SHEET = "per_scenario"
GT_ROLES_PER_IMG_SHEET = "per_img"
GT_ROLE_PER_SCENARIO_XLSX = GT_ROLES_XLSX  # alias; sheet per_scenario
ROLE_MATCH_MAX_DIST_M = 15.0
OBJECTS_MATCHES_XLSX = GT_VISUAL_DIR / "objects_matches.xlsx"
GT_MATCHES_XLSX = OBJECTS_MATCHES_XLSX  # alias
GT_YAML = GT_LABELS_DIR / "data.yaml"
POSE_DIR = REPO_ROOT / "perception" / "vision" / "input" / "telemetry"
DEFAULT_MANIFEST = _EVAL_DIR / "evaluation_manifest.csv"
VISUAL_PERCEPTION_EVAL_DIR = _EVAL_DIR / "results" / "visual_perception_eval"
SLS_EVAL_DIR = _EVAL_DIR / "results" / "sls_eval"
AUDIO_PERCEPTION_EVAL_DIR = _EVAL_DIR / "results" / "audio_perception_eval"
DETECTION_REPORT_XLSX = VISUAL_PERCEPTION_EVAL_DIR / "detection_metrics.xlsx"
LOCALIZATION_REPORT_XLSX = VISUAL_PERCEPTION_EVAL_DIR / "localization_metrics.xlsx"
TRIANGULATION_LOCALIZATION_REPORT_XLSX = (
    VISUAL_PERCEPTION_EVAL_DIR / "triangulation_localization_metrics.xlsx"
)
MULTI_UAV_COUNT_REPORT_XLSX = VISUAL_PERCEPTION_EVAL_DIR / "multi_uav_count_metrics.xlsx"
SCENE_COUNT_REPORT_XLSX = SLS_EVAL_DIR / "scene_count_metrics.xlsx"
COMM_DEMAND_COMPLIANCE_REPORT_XLSX = SLS_EVAL_DIR / "communication_demand_compliance.xlsx"
ROLE_ASSIGNMENT_REPORT_XLSX = SLS_EVAL_DIR / "role_assignment_metrics.xlsx"
OBJECT_MATCHING_REPORT_XLSX = VISUAL_PERCEPTION_EVAL_DIR / "object_matching_metrics.xlsx"
SPEAKER_COUNT_XLSX = GT_AUDIO_DIR / "speaker_count.xlsx"
AUDIO_CUES_XLSX = GT_AUDIO_DIR / "audio_cues.xlsx"
SPEAKER_COUNT_REPORT_XLSX = AUDIO_PERCEPTION_EVAL_DIR / "speaker_count_metrics.xlsx"

# counts_after_match keys (fusion) ↔ objects_count.csv columns
COUNT_CLASS_NAMES = ("person", "vehicle", "emergency_vehicle")
MEAN_RELATIVE_COUNT_ERROR_COL = "Relative mean count error"
GT_COUNT_COLUMN = {
    "person": "Person",
    "vehicle": "Vehicle",
    "emergency_vehicle": "Emergency_vehicle",
}
FUSION_MATCHING_DIR = REPO_ROOT / "fusion" / "matching"
LOC_IMAGE_W = 1280
LOC_IMAGE_H = 720
LOC_FOV_X = 90.0

EVAL_CLASS_IDS = (0, 1, 2)
PRED_CLASS_TO_EVAL_ID = {
    "person": 0,
    "normal_vehicle": 1,
    "emergency_vehicle": 2,
}
EXCEL_CLASS_TO_EVAL_ID = {
    "person": 0,
    "normal_vehicle": 1,
    "emergency_vehicle": 2,
    "vehicle": 1,
}
EXCEL_ALIGN_MAX_DIST_M = 50.0
IOU_THRESH = 0.5

LIGHTING_EN = {
    "tarde": "afternoon",
    "noite": "night",
    "dia": "day",
}


@dataclass(frozen=True)
class Box:
    cls_id: int
    xc: float
    yc: float
    w: float
    h: float
    conf: float = 1.0

    def xyxy(self) -> Tuple[float, float, float, float]:
        x1 = self.xc - self.w / 2.0
        y1 = self.yc - self.h / 2.0
        x2 = self.xc + self.w / 2.0
        y2 = self.yc + self.h / 2.0
        return x1, y1, x2, y2


@dataclass
class EvalContext:
    class_names: Dict[int, str]
    img_lighting: Dict[int, str]
    gt_files: Dict[int, Path]
    image_ids: List[int]
    pred_paths: Dict[int, Path]


@dataclass(frozen=True)
class MonoUavRun:
    run_id: str
    scenario_num: int
    view_id: str
    img_ref: int
    visual_json: Path
    sls_json: Optional[Path]


@dataclass(frozen=True)
class MultiUavRun:
    run_id: str
    scenario_num: int
    view_ref_id: str
    view_other_id: str
    img_ref: int
    visual_ref: Path
    visual_other: Path
    cross_view_json: Optional[Path]
    sls_json: Optional[Path]


def build_img_metadata() -> Dict[int, str]:
    lighting: Dict[int, str] = {}
    for spec in SCENARIOS:
        for group in spec.groups:
            en = LIGHTING_EN.get(group.name, group.name)
            for n in group.image_ids:
                lighting[n] = en
    return lighting


def load_class_names() -> Dict[int, str]:
    with open(GT_YAML, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data["names"]
    if isinstance(names, list):
        return {i: n for i, n in enumerate(names)}
    return {int(k): v for k, v in names.items()}


def discover_gt_label_files() -> Dict[int, Path]:
    out: Dict[int, Path] = {}
    for path in GT_LABELS_DIR.glob("*.txt"):
        m = re.match(r"^(\d+)_", path.name)
        if m:
            out[int(m.group(1))] = path
    return dict(sorted(out.items()))


def visual_paths_from_manifest(manifest_path: Path) -> Dict[int, Path]:
    by_img: Dict[int, Path] = {}
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("multi_uav", "")).lower() in ("true", "1", "yes"):
                continue
            img_col = row.get("view1_image_num") or row.get("image_nums", "")
            if not img_col:
                continue
            img_num = int(str(img_col).split(",")[0].strip())
            rel = row.get("view1_visual_pred_path", "").strip()
            if not rel:
                continue
            path = REPO_ROOT / rel.replace("\\", "/")
            if img_num not in by_img:
                by_img[img_num] = path
    return by_img


def resolve_system_predictions(
    image_ids: Sequence[int],
    manifest_path: Path,
) -> Tuple[Dict[int, Path], List[int]]:
    manifest_paths = visual_paths_from_manifest(manifest_path) if manifest_path.is_file() else {}
    resolved: Dict[int, Path] = {}
    missing: List[int] = []

    for img_num in image_ids:
        cache = VISION_CACHE / f"{img_num:05d}.json"
        if cache.is_file():
            resolved[img_num] = cache
            continue
        manifest_p = manifest_paths.get(img_num)
        if manifest_p and manifest_p.is_file():
            resolved[img_num] = manifest_p
            continue
        missing.append(img_num)

    return resolved, missing


def load_eval_context(manifest_path: Path) -> EvalContext:
    gt_files = discover_gt_label_files()
    image_ids = sorted(gt_files.keys())
    pred_paths, missing = resolve_system_predictions(image_ids, manifest_path)
    print(f"Ground truth images: {len(image_ids)}")
    print(f"System predictions found: {len(pred_paths)}")
    if missing:
        print(f"Missing predictions: {len(missing)} (e.g. img{missing[0]}, ...)")
        raise FileNotFoundError(
            f"{len(missing)} images have no system visual.json. "
            "Run: python scripts/run_all_combinations.py --steps vision"
        )
    return EvalContext(
        class_names=load_class_names(),
        img_lighting=build_img_metadata(),
        gt_files=gt_files,
        image_ids=image_ids,
        pred_paths=pred_paths,
    )


def parse_yolo_label_file(path: Path, *, default_conf: float = 1.0) -> List[Box]:
    boxes: List[Box] = []
    text = path.read_text(encoding="utf-8-sig")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        cls_id = int(float(parts[0]))
        xc, yc, w, h = map(float, parts[1:5])
        if cls_id not in EVAL_CLASS_IDS:
            continue
        boxes.append(Box(cls_id=cls_id, xc=xc, yc=yc, w=w, h=h, conf=default_conf))
    return boxes


def parse_meters(value: object) -> float:
    return float(str(value).split()[0])


def xy_from_position_field(pos: Optional[dict]) -> Optional[Tuple[float, float]]:
    if not pos:
        return None
    x_raw, y_raw = pos.get("x"), pos.get("y")
    if x_raw is None or y_raw is None:
        return None
    return parse_meters(x_raw), parse_meters(y_raw)


def boxes_by_object_id(payload: dict) -> Dict[int, Box]:
    out: Dict[int, Box] = {}
    for obj in payload.get("objects", []):
        cls_name = obj.get("class", "")
        cls_id = PRED_CLASS_TO_EVAL_ID.get(cls_name)
        if cls_id is None:
            continue
        bb = obj.get("bbox")
        if not bb:
            continue
        conf_raw = obj.get("detection_confidence", "1.0")
        conf = float(str(conf_raw).split()[0])
        out[int(obj["id"])] = Box(
            cls_id=cls_id,
            xc=float(bb["xc"]),
            yc=float(bb["yc"]),
            w=float(bb["w"]),
            h=float(bb["h"]),
            conf=conf,
        )
    return out


def _sls_path_from_manifest_row(row: dict) -> Optional[Path]:
    rel = str(row.get("run_dir", "")).strip()
    if not rel:
        return None
    path = REPO_ROOT / rel.replace("\\", "/") / "fusion" / "sls.json"
    return path if path.is_file() else None


def iter_mono_uav_runs(manifest_path: Path) -> Iterator[MonoUavRun]:
    """Manifest rows with one image + audio (multi_uav=false, e.g. imgN_audM)."""
    if not manifest_path.is_file():
        return
    with open(manifest_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if str(row.get("multi_uav", "")).lower() in ("true", "1", "yes"):
                continue
            visual_rel = row.get("view1_visual_pred_path", "").strip()
            if not visual_rel:
                continue
            visual_path = REPO_ROOT / visual_rel.replace("\\", "/")
            if not visual_path.is_file():
                continue
            img_ref = int(str(row.get("view1_image_num", "")).strip())
            scenario_num = parse_scenario_number(row.get("scenario"))
            if scenario_num is None:
                continue
            yield MonoUavRun(
                run_id=str(row.get("run_id", "")),
                scenario_num=scenario_num,
                view_id=str(row.get("view1_view_id", "")),
                img_ref=img_ref,
                visual_json=visual_path,
                sls_json=_sls_path_from_manifest_row(row),
            )


def iter_multi_uav_runs(manifest_path: Path) -> Iterator[MultiUavRun]:
    """Manifest rows with two views (multi_uav=true)."""
    if not manifest_path.is_file():
        return
    with open(manifest_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if str(row.get("multi_uav", "")).lower() not in ("true", "1", "yes"):
                continue
            v1 = row.get("view1_visual_pred_path", "").strip()
            v2 = row.get("view2_visual_pred_path", "").strip()
            if not v1 or not v2:
                continue
            p1 = REPO_ROOT / v1.replace("\\", "/")
            p2 = REPO_ROOT / v2.replace("\\", "/")
            if not p1.is_file() or not p2.is_file():
                continue
            img_ref = int(str(row.get("view1_image_num", "")).strip())
            scenario_num = parse_scenario_number(row.get("scenario"))
            if scenario_num is None:
                continue
            cv = row.get("cross_view_path", "").strip()
            cv_path = REPO_ROOT / cv.replace("\\", "/") if cv else None
            yield MultiUavRun(
                run_id=str(row.get("run_id", "")),
                scenario_num=scenario_num,
                view_ref_id=str(row.get("view1_view_id", "")),
                view_other_id=str(row.get("view2_view_id", "")),
                img_ref=img_ref,
                visual_ref=p1,
                visual_other=p2,
                cross_view_json=cv_path if cv_path and cv_path.is_file() else None,
                sls_json=_sls_path_from_manifest_row(row),
            )


def load_speaker_count_gt(path: Path | None = None) -> pd.DataFrame:
    path = path or SPEAKER_COUNT_XLSX
    if not path.is_file():
        raise FileNotFoundError(f"Speaker count GT not found: {path}")
    df = pd.read_excel(path, sheet_name=0)
    required = {"audio", "speaker_count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"speaker_count.xlsx missing columns: {missing}")
    return df


def predicted_speaker_count_from_transcript(transcript_path: Path) -> Optional[int]:
    data = json.loads(transcript_path.read_text(encoding="utf-8"))
    analytics = data.get("analytics") or {}
    raw = analytics.get("speaker_count", data.get("speaker_count"))
    if raw is not None:
        return int(raw)
    speakers = {
        str(s.get("speaker"))
        for s in data.get("segments", [])
        if s.get("speaker")
    }
    return len(speakers) if speakers else None


def iter_audio_transcripts_from_manifest(
    manifest_path: Path,
) -> Iterator[Dict[str, Any]]:
    """
    One row per unique audio_stem in the manifest (same transcript for all views).
    Yields audio_stem, audio_file, scenario, transcript_path, run_id.
    """
    if not manifest_path.is_file():
        return
    seen: Set[str] = set()
    with open(manifest_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            stem = str(row.get("audio_stem", "")).strip()
            if not stem or stem in seen:
                continue
            rel = str(row.get("transcript_path", "")).strip()
            if not rel:
                continue
            transcript_path = REPO_ROOT / rel.replace("\\", "/")
            if not transcript_path.is_file():
                continue
            seen.add(stem)
            yield {
                "audio_stem": stem,
                "audio_file": f"{stem}.wav",
                "scenario": str(row.get("scenario", "")),
                "transcript_path": transcript_path,
                "run_id": str(row.get("run_id", "")),
            }


def cross_view_result_for_run(run: MultiUavRun) -> dict:
    """Full cross-view payload (read fusion/cross_view.json or compute)."""
    if run.cross_view_json is not None:
        return json.loads(run.cross_view_json.read_text(encoding="utf-8"))

    visual_ref = json.loads(run.visual_ref.read_text(encoding="utf-8"))
    visual_other = json.loads(run.visual_other.read_text(encoding="utf-8"))
    if str(FUSION_MATCHING_DIR) not in sys.path:
        sys.path.insert(0, str(FUSION_MATCHING_DIR))
    from cross_view_match import run_cross_view_match  # noqa: WPS433

    return run_cross_view_match(
        [(run.view_ref_id, visual_ref), (run.view_other_id, visual_other)]
    )


def cross_view_matches_for_run(run: MultiUavRun) -> List[dict]:
    return list(cross_view_result_for_run(run).get("matches") or [])


def parse_scenario_number(scenario: object) -> Optional[int]:
    if scenario is None or (isinstance(scenario, float) and pd.isna(scenario)):
        return None
    text = str(scenario).strip().lower()
    if text.isdigit():
        return int(text)
    match = re.match(r"scenario\s*(\d+)", text)
    if match:
        return int(match.group(1))
    return None


def _read_count_table(path: Path, *, id_column: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    if path.suffix.lower() in (".xlsx", ".xls"):
        xl = pd.ExcelFile(path)
        sheet = sheet_name if sheet_name and sheet_name in xl.sheet_names else xl.sheet_names[0]
        df = pd.read_excel(path, sheet_name=sheet)
    else:
        df = pd.read_csv(path, sep=";")
    return df.rename(columns=lambda c: str(c).strip())


def load_objects_count_gt(path: Path | None = None) -> Dict[int, Dict[str, int]]:
    """Per-image GT counts from objects_count.xlsx sheet 'objects_count' (or .csv)."""
    if path is not None:
        gt_path = path
        sheet = GT_OBJECTS_COUNT_SHEET if gt_path.suffix.lower() in (".xlsx", ".xls") else None
    elif GT_OBJECTS_COUNT_CSV.is_file():
        gt_path = GT_OBJECTS_COUNT_CSV
        sheet = None
    elif GT_OBJECTS_COUNT_XLSX.is_file():
        gt_path = GT_OBJECTS_COUNT_XLSX
        sheet = GT_OBJECTS_COUNT_SHEET
    else:
        raise FileNotFoundError(
            f"Ground-truth counts not found: {GT_OBJECTS_COUNT_CSV} or {GT_OBJECTS_COUNT_XLSX}"
        )
    df = _read_count_table(gt_path, id_column="Img", sheet_name=sheet)
    if "Img" not in df.columns:
        raise ValueError(f"{gt_path.name} must have an 'Img' column")
    out: Dict[int, Dict[str, int]] = {}
    for _, row in df.iterrows():
        if pd.isna(row.get("Img")):
            continue
        img_num = int(row["Img"])
        out[img_num] = {
            name: int(row[col])
            for name, col in GT_COUNT_COLUMN.items()
            if col in df.columns and not pd.isna(row[col])
        }
    return out


def load_scenario_count_gt(path: Path | None = None) -> Dict[int, Dict[str, int]]:
    """Per-scenario GT counts from objects_count.xlsx sheet 'per_scenario'."""
    gt_path = path or GT_OBJECTS_COUNT_XLSX
    if not gt_path.is_file():
        raise FileNotFoundError(f"Scenario count GT not found: {gt_path}")
    df = _read_count_table(gt_path, id_column="Scenario", sheet_name=GT_SCENARIO_COUNT_SHEET)
    scenario_col = None
    for col in df.columns:
        if col.strip().lower() == "scenario":
            scenario_col = col
            break
    if scenario_col is None:
        raise ValueError(f"{gt_path.name} sheet '{GT_SCENARIO_COUNT_SHEET}' must have a Scenario column")
    out: Dict[int, Dict[str, int]] = {}
    for _, row in df.iterrows():
        if pd.isna(row.get(scenario_col)):
            continue
        scenario_num = int(row[scenario_col])
        out[scenario_num] = {
            name: int(row[col])
            for name, col in GT_COUNT_COLUMN.items()
            if col in df.columns and not pd.isna(row[col])
        }
    return out


def load_role_scenario_gt(path: Path | None = None) -> Dict[int, List[dict]]:
    """Per-scenario person roles and positions (gt_roles.xlsx sheet per_scenario)."""
    gt_path = path or GT_ROLES_XLSX
    if not gt_path.is_file():
        raise FileNotFoundError(f"Role GT not found: {gt_path}")
    df = pd.read_excel(gt_path, sheet_name=GT_ROLES_PER_SCENARIO_SHEET, header=0)
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = {"scenario", "classe", "role", "x (m)", "y (m)"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{gt_path.name} missing columns: {sorted(missing)}")
    out: Dict[int, List[dict]] = defaultdict(list)
    for _, row in df.iterrows():
        if pd.isna(row.get("scenario")):
            continue
        if str(row.get("classe", "")).strip().lower() != "person":
            continue
        scenario_num = int(row["scenario"])
        z_val = row.get("z (m)")
        out[scenario_num].append(
            {
                "role": str(row["role"]).strip().lower(),
                "x": float(row["x (m)"]),
                "y": float(row["y (m)"]),
                "z": float(z_val) if not pd.isna(z_val) else 0.0,
            }
        )
    return dict(out)


def load_gt_roles_per_img(path: Path | None = None) -> Dict[Tuple[str, int], str]:
    """Per-image person roles keyed by (img, visual.json object id)."""
    gt_path = path or GT_ROLES_XLSX
    if not gt_path.is_file():
        raise FileNotFoundError(f"Role GT not found: {gt_path}")
    df = pd.read_excel(gt_path, sheet_name=GT_ROLES_PER_IMG_SHEET, header=0)
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = {"img", "id", "gt_role"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{gt_path.name} sheet '{GT_ROLES_PER_IMG_SHEET}' missing: {sorted(missing)}")
    out: Dict[Tuple[str, int], str] = {}
    for _, row in df.iterrows():
        if pd.isna(row.get("img")) or pd.isna(row.get("id")):
            continue
        img = str(row["img"]).strip()
        oid = int(row["id"])
        role = str(row["gt_role"]).strip().lower()
        out[(img, oid)] = role
    return out


def fused_counts_from_cross_view(payload: dict) -> Optional[Dict[str, int]]:
    """counts_after_match: unique entities (matched clusters + unmatched) per class."""
    raw = payload.get("counts_after_match")
    if not raw:
        return None
    out = {name: int(raw.get(name, 0)) for name in COUNT_CLASS_NAMES}
    if out["vehicle"] == 0 and raw.get("normal_vehicle"):
        out["vehicle"] = int(raw["normal_vehicle"])
    return out


def counts_from_visual_json(payload: dict) -> Dict[str, int]:
    out = {name: 0 for name in COUNT_CLASS_NAMES}
    for obj in payload.get("objects", []):
        cls_name = str(obj.get("class", ""))
        if cls_name == "person":
            out["person"] += 1
        elif cls_name == "normal_vehicle":
            out["vehicle"] += 1
        elif cls_name == "emergency_vehicle":
            out["emergency_vehicle"] += 1
    return out


def _audio_only_counts(payload: dict) -> Dict[str, int]:
    """Objects explicitly tagged audio_only in SLS (radio gap-fill)."""
    out = {name: 0 for name in COUNT_CLASS_NAMES}
    for obj in payload.get("objects", []):
        if not obj.get("audio_only"):
            continue
        cls = str(obj.get("class", ""))
        if cls == "person":
            out["person"] += 1
        elif cls == "normal_vehicle":
            out["vehicle"] += 1
        elif cls == "emergency_vehicle":
            out["emergency_vehicle"] += 1
    return out


def sls_audio_only_deltas(payload: dict) -> Dict[str, int]:
    """audio_only counts by eval class (radio gap-fill; not scene GT)."""
    return _audio_only_counts(payload)


def counts_vision_and_audio(
    vision_only: Dict[str, int],
    sls_payload: dict,
) -> Dict[str, int]:
    """
    Vision+audio = vision-only baseline + ``audio_only`` gap-fill from SLS.

    Audio only adds entities not already in the visual baseline (never removes).
    """
    audio = _audio_only_counts(sls_payload)
    return {
        name: int(vision_only.get(name, 0)) + int(audio.get(name, 0))
        for name in COUNT_CLASS_NAMES
    }


def sls_reconciles_with_baseline(
    vision_only: Dict[str, int],
    sls_payload: dict,
) -> bool:
    """True when counts_by_class == vision_only baseline + audio_only objects."""
    expected = {
        name: int(vision_only.get(name, 0)) + int(_audio_only_counts(sls_payload).get(name, 0))
        for name in COUNT_CLASS_NAMES
    }
    actual = counts_from_sls(sls_payload)
    return all(actual[name] == expected[name] for name in COUNT_CLASS_NAMES)


def counts_from_sls(payload: dict) -> Dict[str, int]:
    raw = payload.get("counts_by_class") or {}
    return {
        "person": int(raw.get("person", 0) or 0),
        "vehicle": int(raw.get("normal_vehicle", 0) or 0),
        "emergency_vehicle": int(raw.get("emergency_vehicle", 0) or 0),
    }


def counts_from_sls_for_scene_gt(payload: dict) -> Dict[str, int]:
    """
    SLS counts aligned with objects_count.xlsx (visible scene entities).

    - person: all persons in SLS (including audio_only gap-fill for occluded civilians)
    - vehicles: only visually grounded objects (exclude audio_only vehicle placeholders
      created to match radio mentions that exceed the labeled image counts)
    """
    by_class = counts_from_sls(payload)
    for obj in payload.get("objects", []):
        if not obj.get("audio_only"):
            continue
        cls = str(obj.get("class", ""))
        if cls == "normal_vehicle":
            by_class["vehicle"] = max(0, by_class["vehicle"] - 1)
        elif cls == "emergency_vehicle":
            by_class["emergency_vehicle"] = max(0, by_class["emergency_vehicle"] - 1)
    return by_class


def read_raw_eval_labels(path: Path) -> List[Tuple[int, float, float, float, float]]:
    rows: List[Tuple[int, float, float, float, float]] = []
    text = path.read_text(encoding="utf-8-sig")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        cls_id = int(float(parts[0]))
        if cls_id not in EVAL_CLASS_IDS:
            continue
        xc, yc, w, h = map(float, parts[1:5])
        rows.append((cls_id, xc, yc, w, h))
    return rows


def gt_boxes_and_xy(img_num: int, label_path: Path) -> Tuple[List[Box], List[Optional[Tuple[float, float]]]]:
    raw = read_raw_eval_labels(label_path)
    boxes = [Box(cls_id=c, xc=xc, yc=yc, w=w, h=h) for c, xc, yc, w, h in raw]
    if not raw:
        return boxes, []

    telemetry = POSE_DIR / f"{img_num:05d}.txt"
    if not telemetry.is_file():
        return boxes, [None] * len(boxes)

    vision_code = REPO_ROOT / "perception" / "vision" / "code"
    if str(vision_code) not in sys.path:
        sys.path.insert(0, str(vision_code))
    from integration_pipeline import localize_objects_with_confidence_from_labels  # noqa: WPS433

    localized = localize_objects_with_confidence_from_labels(
        raw,
        str(telemetry),
        W=LOC_IMAGE_W,
        H=LOC_IMAGE_H,
        fov_x=LOC_FOV_X,
    )
    id_to_xy: Dict[int, Tuple[float, float]] = {}
    for obj in localized:
        if obj.get("class") == "fire":
            continue
        pos = obj["position"]
        id_to_xy[int(obj["id"])] = (float(pos[0]), float(pos[1]))

    xy: List[Optional[Tuple[float, float]]] = [id_to_xy.get(i) for i in range(len(raw))]
    return boxes, xy


def view_to_img_num(view_id: str) -> int:
    return int(re.sub(r"^img", "", str(view_id).strip(), flags=re.I))


def normalize_view_pair(view_a: str, view_b: str) -> Tuple[str, str]:
    va, vb = str(view_a).strip(), str(view_b).strip()
    if view_to_img_num(va) <= view_to_img_num(vb):
        return va, vb
    return vb, va


def canonical_object_match(
    view_a: str,
    local_id_a: int,
    view_b: str,
    local_id_b: int,
) -> Tuple[str, int, str, int]:
    """(view_a, id_a, view_b, id_b) with lower image index first."""
    va, vb = normalize_view_pair(view_a, view_b)
    la, lb = int(local_id_a), int(local_id_b)
    if va == str(view_a).strip():
        return va, la, vb, lb
    return va, lb, vb, la


def object_match_entity_ids(
    view_a: str,
    local_id_a: int,
    view_b: str,
    local_id_b: int,
) -> Tuple[int, int]:
    m = canonical_object_match(view_a, local_id_a, view_b, local_id_b)
    return m[1], m[3]


def pair_key_from_row(
    scenario: str,
    group: str,
    view_a: str,
    view_b: str,
) -> Tuple[str, str, str, str]:
    va, vb = normalize_view_pair(view_a, view_b)
    return str(scenario), str(group), va, vb


def load_gt_matches_df(path: Path | None = None) -> pd.DataFrame:
    path = path or OBJECTS_MATCHES_XLSX
    if not path.is_file():
        raise FileNotFoundError(f"Ground-truth matches not found: {path}")
    try:
        df = pd.read_excel(path, sheet_name="objects_matches")
    except ValueError:
        df = pd.read_excel(path, sheet_name="gt_matches")  # legacy sheet name
    required = {"scenario", "group", "view_a", "view_b", "local_id_a", "local_id_b"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"objects_matches.xlsx missing columns: {missing}")
    return df.dropna(subset=list(required)).copy()


def build_gt_matches_by_pair(
    gt_df: pd.DataFrame,
) -> Dict[Tuple[str, str, str, str], set[Tuple[int, int]]]:
    out: Dict[Tuple[str, str, str, str], set[Tuple[int, int]]] = defaultdict(set)
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
        out[pk].add(ent)
    return dict(out)


def cross_view_entity_from_match(m: dict) -> Tuple[int, int]:
    return object_match_entity_ids(
        str(m.get("view_a", "")),
        int(m["id_a"]),
        str(m.get("view_b", "")),
        int(m["id_b"]),
    )


def pred_xy_by_object_id(payload: dict) -> Dict[int, Tuple[float, float]]:
    out: Dict[int, Tuple[float, float]] = {}
    for obj in payload.get("objects", []):
        oid = obj.get("id")
        if oid is None:
            continue
        xy = xy_from_position_field(obj.get("position"))
        if xy is not None:
            out[int(oid)] = xy
    return out


def gt_xy_from_objects_position(
    positions_df: pd.DataFrame,
    view_id: str,
    obj_id: int,
) -> Optional[Tuple[float, float]]:
    rows = positions_df[
        (positions_df["img"].astype(str) == str(view_id))
        & (positions_df["obj_id"] == int(obj_id))
    ]
    if rows.empty:
        return None
    r = rows.iloc[0]
    return float(r["x (m)"]), float(r["y (m)"])


def gt_xy_for_matched_entity(
    positions_df: pd.DataFrame,
    view_a: str,
    local_id_a: int,
    view_b: str,
    local_id_b: int,
) -> Optional[Tuple[float, float]]:
    """
    GT XY (m) for a confirmed cross-view match.

    Uses objects_position at the reference view (lower image index, view_a after
    normalization) and local_id_a — same ids as objects_matches.xlsx. Falls back to
    view_b if the reference row is missing.
    """
    va, ia, vb, ib = canonical_object_match(view_a, local_id_a, view_b, local_id_b)
    return gt_xy_from_objects_position(positions_df, va, ia) or gt_xy_from_objects_position(
        positions_df, vb, ib
    )


def gt_xy_spread_between_views(
    positions_df: pd.DataFrame,
    view_a: str,
    local_id_a: int,
    view_b: str,
    local_id_b: int,
) -> Optional[float]:
    """Distance (m) between per-view GT positions in objects_position (diagnostic)."""
    va, ia, vb, ib = canonical_object_match(view_a, local_id_a, view_b, local_id_b)
    xy_a = gt_xy_from_objects_position(positions_df, va, ia)
    xy_b = gt_xy_from_objects_position(positions_df, vb, ib)
    if not xy_a or not xy_b:
        return None
    return float(np.hypot(xy_a[0] - xy_b[0], xy_a[1] - xy_b[1]))


def build_cross_view_preds_by_pair(
    manifest_path: Path,
) -> Dict[Tuple[str, str, str, str], dict]:
    """
    Best multi-UAV run per (scenario, group, view_a, view_b): most predicted matches.
    Value: {matches, run_id, cross_view_path, scenario, group, view_a, view_b}.
    """
    from fusion.matching.object_matching_core import is_excluded_match_class

    def object_class_in_visual(visuals: dict, view_id: str, obj_id: int) -> str:
        for obj in visuals.get(view_id, {}).get("objects", []):
            if int(obj.get("id", -1)) == int(obj_id):
                return str(obj.get("class", "")).strip()
        return ""

    def should_include(m: dict, visuals: dict) -> bool:
        va = str(m.get("view_a", ""))
        vb = str(m.get("view_b", ""))
        ia, ib = int(m["id_a"]), int(m["id_b"])
        known_a = {
            int(o["id"])
            for o in visuals.get(va, {}).get("objects", [])
            if o.get("id") is not None
        }
        known_b = {
            int(o["id"])
            for o in visuals.get(vb, {}).get("objects", [])
            if o.get("id") is not None
        }
        if known_a and ia not in known_a:
            return False
        if known_b and ib not in known_b:
            return False
        ca = object_class_in_visual(visuals, va, ia)
        cb = object_class_in_visual(visuals, vb, ib)
        return not is_excluded_match_class(ca) and not is_excluded_match_class(cb)

    by_pair: Dict[Tuple[str, str, str, str], dict] = {}
    if not manifest_path.is_file():
        return by_pair

    with open(manifest_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if str(row.get("multi_uav", "")).lower() not in ("true", "1", "yes"):
                continue
            cv_rel = row.get("cross_view_path", "").strip()
            if not cv_rel:
                continue
            cv_path = REPO_ROOT / cv_rel.replace("\\", "/")
            if not cv_path.is_file():
                continue
            scenario = str(row["scenario"])
            group = str(row["group"])
            v1 = str(row.get("view1_view_id", ""))
            v2 = str(row.get("view2_view_id", ""))
            if not v1 or not v2:
                continue
            pk = pair_key_from_row(scenario, group, v1, v2)
            payload = json.loads(cv_path.read_text(encoding="utf-8"))
            visuals = payload.get("visuals") or {}
            matches = [m for m in (payload.get("matches") or []) if should_include(m, visuals)]
            entry = {
                "scenario": scenario,
                "group": group,
                "view_a": pk[2],
                "view_b": pk[3],
                "matches": matches,
                "run_id": str(row.get("run_id", "")),
                "cross_view_path": cv_path,
            }
            if pk not in by_pair or len(matches) > len(by_pair[pk].get("matches", [])):
                by_pair[pk] = entry
    return by_pair


def load_objects_position_xlsx(path: Path | None = None) -> pd.DataFrame:
    """Ground-truth 3D positions (meters) from groundTruth_data/visual/objects_position.xlsx.

    Supports legacy column ``classe`` or split columns ``true classe`` (eval GT class)
    and ``assigned classe`` (label id used in the pipeline when it differs).
    """
    path = path or GT_POSITIONS_XLSX
    if not path.is_file():
        raise FileNotFoundError(f"Ground-truth positions not found: {path}")
    df = pd.read_excel(path, sheet_name=0)
    if "classe" not in df.columns:
        if "true classe" in df.columns:
            df = df.copy()
            df["classe"] = df["true classe"]
        else:
            raise ValueError(
                "objects_position.xlsx missing class column: need 'classe' or 'true classe'"
            )
    required = {"img", "obj_id", "classe", "x (m)", "y (m)"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"objects_position.xlsx missing columns: {missing}")
    return df


def _excel_row_class_id(classe: object) -> Optional[int]:
    return EXCEL_CLASS_TO_EVAL_ID.get(str(classe).strip())


def gt_xy_from_excel_for_boxes(
    img_num: int,
    gt_boxes: Sequence[Box],
    positions_df: pd.DataFrame,
    label_path: Path,
    *,
    align_max_dist_m: float = EXCEL_ALIGN_MAX_DIST_M,
) -> List[Optional[Tuple[float, float]]]:
    """
    Map each GT bbox index to (x, y) in meters from objects_position.xlsx.

    Bbox indices come from gt_labels; Excel rows are aligned per class using
    geometry only as a matching hint (metrics use Excel coordinates).
    """
    n = len(gt_boxes)
    xy: List[Optional[Tuple[float, float]]] = [None] * n
    img_key = f"img{img_num}"
    rows = positions_df[positions_df["img"].astype(str) == img_key]
    if rows.empty:
        return xy

    _, geo_hint = gt_boxes_and_xy(img_num, label_path)

    for cls_id in EVAL_CLASS_IDS:
        excel_cls = rows[rows["classe"].map(_excel_row_class_id) == cls_id]
        gt_inds = [i for i, b in enumerate(gt_boxes) if b.cls_id == cls_id]
        if excel_cls.empty or not gt_inds:
            continue

        used: set[int] = set()
        for _, erow in excel_cls.iterrows():
            ex = np.array([float(erow["x (m)"]), float(erow["y (m)"])])
            best_i: Optional[int] = None
            best_d = align_max_dist_m + 1.0
            for gi in gt_inds:
                if gi in used:
                    continue
                hint = geo_hint[gi] if gi < len(geo_hint) else None
                if hint is None:
                    continue
                d = float(np.linalg.norm(np.array(hint) - ex))
                if d < best_d:
                    best_d = d
                    best_i = gi
            if best_i is not None and best_d <= align_max_dist_m:
                used.add(best_i)
                xy[best_i] = (float(erow["x (m)"]), float(erow["y (m)"]))

    return xy


def boxes_from_visual_json(payload: dict) -> List[Box]:
    boxes: List[Box] = []
    for obj in payload.get("objects", []):
        cls_name = obj.get("class", "")
        cls_id = PRED_CLASS_TO_EVAL_ID.get(cls_name)
        if cls_id is None:
            continue
        bb = obj.get("bbox")
        if not bb:
            continue
        conf_raw = obj.get("detection_confidence", "1.0")
        conf = float(str(conf_raw).split()[0])
        boxes.append(
            Box(
                cls_id=cls_id,
                xc=float(bb["xc"]),
                yc=float(bb["yc"]),
                w=float(bb["w"]),
                h=float(bb["h"]),
                conf=conf,
            )
        )
    return boxes


def boxes_and_xy_from_visual_json(payload: dict) -> Tuple[List[Box], List[Optional[Tuple[float, float]]]]:
    boxes = boxes_from_visual_json(payload)
    xy: List[Optional[Tuple[float, float]]] = []
    for obj in payload.get("objects", []):
        cls_name = obj.get("class", "")
        if PRED_CLASS_TO_EVAL_ID.get(cls_name) is None:
            continue
        pos = obj.get("position")
        bb = obj.get("bbox")
        if pos and bb:
            xy.append((parse_meters(pos["x"]), parse_meters(pos["y"])))
        else:
            xy.append(None)
    return boxes, xy


def box_iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a.xyxy()
    bx1, by1, bx2, by2 = b.xyxy()
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_image(
    gt: Sequence[Box], pred: Sequence[Box]
) -> Tuple[int, int, int, List[Tuple[float, bool]]]:
    gt_by_cls: Dict[int, List[Box]] = defaultdict(list)
    pred_by_cls: Dict[int, List[Box]] = defaultdict(list)
    for b in gt:
        gt_by_cls[b.cls_id].append(b)
    for b in pred:
        pred_by_cls[b.cls_id].append(b)

    tp = fp = fn = 0
    det_scores: List[Tuple[float, bool]] = []

    for cls_id in EVAL_CLASS_IDS:
        gts = gt_by_cls.get(cls_id, [])
        preds = sorted(pred_by_cls.get(cls_id, []), key=lambda b: -b.conf)
        matched_gt = [False] * len(gts)

        for p in preds:
            best_iou = 0.0
            best_j = -1
            for j, g in enumerate(gts):
                if matched_gt[j]:
                    continue
                iou = box_iou(p, g)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_iou >= IOU_THRESH and best_j >= 0:
                matched_gt[best_j] = True
                tp += 1
                det_scores.append((p.conf, True))
            else:
                fp += 1
                det_scores.append((p.conf, False))

        fn += sum(1 for m in matched_gt if not m)

    return tp, fp, fn, det_scores


def match_true_positive_pairs(gt: Sequence[Box], pred: Sequence[Box]) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for cls_id in EVAL_CLASS_IDS:
        gt_idx = [i for i, b in enumerate(gt) if b.cls_id == cls_id]
        gts = [gt[i] for i in gt_idx]
        preds = sorted(
            [(i, pred[i]) for i in range(len(pred)) if pred[i].cls_id == cls_id],
            key=lambda t: -t[1].conf,
        )
        matched = [False] * len(gts)
        for pi, pbox in preds:
            best_iou = 0.0
            best_j = -1
            for j, gbox in enumerate(gts):
                if matched[j]:
                    continue
                iou = box_iou(pbox, gbox)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_iou >= IOU_THRESH and best_j >= 0:
                matched[best_j] = True
                pairs.append((gt_idx[best_j], pi))
    return pairs


def write_excel(report_path: Path, sheets: Dict[str, pd.DataFrame]) -> Path:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidates = [
        report_path,
        report_path.with_name(f"{report_path.stem}_new{report_path.suffix}"),
        report_path.with_name(f"{report_path.stem}_{stamp}{report_path.suffix}"),
    ]
    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        try:
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                for sheet_name, df in sheets.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            if path.resolve() != report_path.resolve():
                print(
                    f"Warning: could not write {report_path} (file may be open). "
                    f"Wrote {path.name} instead."
                )
            return path
        except PermissionError:
            continue
    raise PermissionError(
        f"Cannot write {report_path.name}. Close it in Excel and re-run."
    )


def average_precision(det_scores: List[Tuple[float, bool]], n_gt: int) -> float:
    if n_gt == 0:
        return float("nan")
    if not det_scores:
        return 0.0
    det_scores = sorted(det_scores, key=lambda x: -x[0])
    tp_cum = fp_cum = 0
    precisions: List[float] = []
    recalls: List[float] = []
    for _conf, is_tp in det_scores:
        if is_tp:
            tp_cum += 1
        else:
            fp_cum += 1
        precisions.append(tp_cum / (tp_cum + fp_cum))
        recalls.append(tp_cum / n_gt)
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        prec_at_rec = [p for p, r in zip(precisions, recalls) if r >= t]
        ap += (max(prec_at_rec) if prec_at_rec else 0.0) / 101.0
    return ap


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def relative_count_error(pred: int, gt: int) -> Optional[float]:
    """Signed relative count error as a fraction of GT: (pred - gt) / gt.

    Returns 0.0 when gt == 0 and pred == 0 (correct absence).
    Returns None when gt == 0 and pred > 0 (undefined relative error).
    """
    if gt == 0:
        return 0.0 if pred == 0 else None
    return (pred - gt) / gt


def mean_relative_count_error(
    rows: Sequence[dict],
    *,
    pred_key: str = "pred_count",
    gt_key: str = "gt_count",
) -> Optional[float]:
    rels: List[float] = []
    for row in rows:
        rel = relative_count_error(int(row[pred_key]), int(row[gt_key]))
        if rel is not None:
            rels.append(rel)
    if not rels:
        return None
    return round(float(np.mean(rels)), 4)


def error_2d_m(
    g_pos: Tuple[float, float], p_pos: Tuple[float, float]
) -> float:
    return float(np.hypot(g_pos[0] - p_pos[0], g_pos[1] - p_pos[1]))


def localization_errors(
    g_pos: Tuple[float, float], p_pos: Tuple[float, float]
) -> Tuple[float, float, float]:
    err_x = abs(g_pos[0] - p_pos[0])
    err_y = abs(g_pos[1] - p_pos[1])
    err_pos = error_2d_m(g_pos, p_pos)
    return err_x, err_y, err_pos


def summarize_localization_metrics(
    errs_x: Sequence[float],
    errs_y: Sequence[float],
    errs_pos: Sequence[float],
) -> Dict[str, Optional[float]]:
    if not errs_pos:
        return {
            "n_pairs": 0,
            "mae_x_m": None,
            "mae_y_m": None,
            "mae_pos_m": None,
        }
    arr_x = np.asarray(errs_x, dtype=float)
    arr_y = np.asarray(errs_y, dtype=float)
    arr_pos = np.asarray(errs_pos, dtype=float)
    valid = np.isfinite(arr_pos)
    if not np.any(valid):
        return {
            "n_pairs": len(errs_pos),
            "mae_x_m": None,
            "mae_y_m": None,
            "mae_pos_m": None,
        }
    return {
        "n_pairs": int(np.sum(valid)),
        "mae_x_m": round(float(np.nanmean(arr_x[valid])), 4),
        "mae_y_m": round(float(np.nanmean(arr_y[valid])), 4),
        "mae_pos_m": round(float(np.nanmean(arr_pos[valid])), 4),
    }


def sorted_lightings(img_lighting: Dict[int, str], image_ids: Sequence[int]) -> List[str]:
    return sorted({img_lighting[i] for i in image_ids})
