#!/usr/bin/env python3
"""
Evaluate LLM/fusion SLS vs evaluation/groundTruth_data/audio/audio_cues.xlsx.

Metrics (single-view runs imgN_audM, median over mono runs):
  - inferred number of people (radio-linked persons in SLS)
  - inferred number of vehicles (explicit counts in radio transcript)
  - role inference (civilian + firefighters mentioned)
  - firefighter role inference (firefighters near the fire)
  - services needed (communications.service_types in SLS)

Output: evaluation/results/audio_perception_eval/audio_cues_metrics.xlsx

Usage (from repo root):
  python evaluation/scripts/evaluate_audio_cues.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from eval_common import (  # noqa: E402
    AUDIO_CUES_XLSX,
    AUDIO_PERCEPTION_EVAL_DIR,
    DEFAULT_MANIFEST,
    REPO_ROOT,
    write_excel,
)

AUDIO_CUES_REPORT_XLSX = AUDIO_PERCEPTION_EVAL_DIR / "audio_cues_metrics.xlsx"

RESPONDER_ROLES = frozenset({"firefighter", "possible_responder"})
MONO_RUN_ID = re.compile(r"^img\d+_aud\d+$", re.IGNORECASE)
RADIO_BASIS_RE = re.compile(
    r"radio|transcript|casualt|civilian|engine|rescue|mentioned|off-scene|firefighter",
    re.IGNORECASE,
)

SERVICE_ALIASES = {
    "thermal image": "thermal_image",
    "thermal imagery": "thermal_image",
    "thermal_image": "thermal_image",
    "image transfer": "image_transfer",
    "image_transfer": "image_transfer",
    "video": "video",
    "voice": "voice",
    "command aggregation": "command_aggregation",
    "command_aggregation": "command_aggregation",
}

METRIC_PEOPLE = "inferred number of people"
METRIC_VEHICLES = "inferred number of vehicles"
METRIC_ROLE_INFERENCE = "role inference"
METRIC_FF_ROLE_INFERENCE = "firefighter role inference"
METRIC_SERVICES = "services needed"

ROLE_LABELS = {
    "civilian": "civilian",
    "firefighter": "firefighter",
    "firefighter_near_fire": "firefighter near fire",
}

PRED_SUFFIX = "(pred)"


def _audio_stem(name: str) -> str:
    return Path(str(name).strip()).stem


def _parse_gt_count(value: object) -> Optional[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip().lower()
    if text in ("---", "", "nan", "none"):
        return None
    if text.isdigit():
        return int(text)
    return None


def _format_roles(roles: Dict[str, int]) -> str:
    if not roles:
        return ""
    return ", ".join(
        f"{ROLE_LABELS.get(k, k)}:{n}" for k, n in sorted(roles.items())
    )


def _build_gt_role_inference(
    civilian: Optional[int],
    firefighters: Optional[int],
) -> Optional[Dict[str, int]]:
    roles: Dict[str, int] = {}
    if civilian is not None:
        roles["civilian"] = civilian
    if firefighters is not None:
        roles["firefighter"] = firefighters
    return roles or None


def _build_gt_ff_role_inference(ff_near_fire: Optional[int]) -> Optional[Dict[str, int]]:
    if ff_near_fire is None:
        return None
    return {"firefighter_near_fire": ff_near_fire}


def _format_pred_role_inference(civilian: int, firefighters: int) -> str:
    parts = []
    if civilian:
        parts.append(f"civilian:{civilian}")
    if firefighters:
        parts.append(f"firefighter:{firefighters}")
    return ", ".join(parts)


def _pred_count_for_role_key(
    key: str,
    *,
    civilian: int,
    firefighters: int,
    ff_near_fire: int,
) -> int:
    if key == "civilian":
        return civilian
    if key == "firefighter_near_fire":
        return ff_near_fire
    if key == "firefighter":
        return firefighters
    return 0


def _match_roles(
    gt_roles: Optional[Dict[str, int]],
    pred_civilian: Optional[int],
    pred_firefighters: Optional[int],
    pred_ff_near_fire: Optional[int],
) -> object:
    if gt_roles is None:
        return pd.NA
    if (
        pred_civilian is None
        or pred_firefighters is None
        or pred_ff_near_fire is None
    ):
        return pd.NA
    for key, expected in gt_roles.items():
        actual = _pred_count_for_role_key(
            key,
            civilian=int(pred_civilian),
            firefighters=int(pred_firefighters),
            ff_near_fire=int(pred_ff_near_fire),
        )
        if actual != expected:
            return False
    return True


def _parse_gt_firefighter_roles(
    value: object,
) -> Tuple[Optional[int], str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None, ""
    text = str(value).strip()
    if text.lower() in ("---", "", "nan", "none"):
        return None, text
    match = re.search(r"(\d+)\s*near\s*the\s*fire", text, re.IGNORECASE)
    if match:
        return int(match.group(1)), text
    if text.isdigit():
        return int(text), text
    return None, text


def _normalize_service_token(token: str) -> str:
    key = token.strip().lower()
    return SERVICE_ALIASES.get(key, key.replace(" ", "_"))


def _parse_gt_services(value: object) -> Optional[Set[str]]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if text.lower() in ("---", "", "nan", "none"):
        return None
    out: Set[str] = set()
    for part in re.split(r"[,;]+", text):
        part = part.strip()
        if part:
            out.add(_normalize_service_token(part))
    return out


def _services_from_sls(data: dict) -> Set[str]:
    comms = data.get("communications") or {}
    raw = comms.get("service_types") or []
    if not isinstance(raw, list):
        raw = [raw]
    return {_normalize_service_token(str(x)) for x in raw if str(x).strip()}


def _parse_gt_vehicles(value: object) -> Dict[str, Optional[int]]:
    out: Dict[str, Optional[int]] = {
        "total": None,
        "emergency": None,
        "min_normal": None,
    }
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return out
    text = str(value).strip()
    if text.lower() in ("---", "", "nan", "none"):
        return out
    em = re.search(r"(\d+)\s*emergency\s+vehicles?", text, re.IGNORECASE)
    if em:
        out["emergency"] = int(em.group(1))
    min_norm = re.search(r">\s*(\d+)\s+vehicles?", text, re.IGNORECASE)
    if min_norm:
        out["min_normal"] = int(min_norm.group(1))
    if text.isdigit():
        out["total"] = int(text)
    elif out["emergency"] is None and out["min_normal"] is None:
        lone = re.match(r"^(\d+)$", text)
        if lone:
            out["total"] = int(lone.group(1))
    return out


def _match_vehicles(
    gt: Dict[str, Optional[int]],
    pred_normal: int,
    pred_emergency: int,
) -> object:
    checks: List[bool] = []
    if gt.get("emergency") is not None:
        checks.append(pred_emergency == gt["emergency"])
    if gt.get("min_normal") is not None:
        checks.append(pred_normal >= gt["min_normal"])
    if gt.get("total") is not None:
        checks.append((pred_normal + pred_emergency) == gt["total"])
    if not checks:
        return pd.NA
    return all(checks)


def _gt_person_count(
    civilians: Optional[int],
    firefighters: Optional[int],
    on_scene_responders: Optional[int],
) -> Optional[int]:
    if civilians is None:
        return None
    if firefighters is not None:
        return civilians + firefighters
    if on_scene_responders is not None and on_scene_responders > 0:
        return civilians + on_scene_responders
    return civilians


def _load_on_scene_responders(transcript_path: Path) -> Optional[int]:
    """Used only to build GT person count when Excel has no firefighters column."""
    if not transcript_path.is_file():
        return None
    data = json.loads(transcript_path.read_text(encoding="utf-8"))
    n = int((data.get("analytics") or {}).get("on_scene_responders_count") or 0)
    return n if n > 0 else None


def _role_has_radio_basis(obj: dict) -> bool:
    if obj.get("audio_only"):
        return True
    basis = obj.get("role_inference_basis") or {}
    if basis.get("source") not in ("transcript", "inference"):
        return False
    return bool(RADIO_BASIS_RE.search(str(basis.get("text", ""))))


def _sls_audio_person_count(data: dict) -> int:
    """Persons in SLS attributed to radio (LLM/fusion), excluding vision-only persons."""
    return sum(
        1
        for obj in data.get("objects", [])
        if obj.get("class") == "person" and _role_has_radio_basis(obj)
    )


def _sls_role_breakdown(data: dict) -> Tuple[int, int, int, str]:
    role_counts: Counter[str] = Counter()
    for obj in data.get("objects", []):
        if obj.get("class") != "person":
            continue
        role = (obj.get("inferred_role") or "unknown_person").lower()
        role_counts[role] += 1
    civilians = role_counts.get("civilian", 0)
    responders = sum(
        n for role, n in role_counts.items() if role in RESPONDER_ROLES
    )
    ff_roles = role_counts.get("firefighter", 0)
    summary = ", ".join(f"{role}:{n}" for role, n in sorted(role_counts.items()))
    return civilians, responders, ff_roles, summary


_WORD_TO_NUM = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}
_OR_MORE_VEHICLES_RE = re.compile(
    r"(?:roughly\s+)?(\d+|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)"
    r"\s+or\s+more\s+vehicles",
    re.IGNORECASE,
)


def _parse_qty_token(token: str) -> Optional[int]:
    if not token:
        return None
    t = token.strip().lower()
    if t.isdigit():
        return int(t)
    return _WORD_TO_NUM.get(t)


def _radio_vehicle_counts(transcript_path: Path) -> Tuple[int, int]:
    """
    Vehicle counts inferred from radio text (peak explicit mention), aligned with
    fusion/llm_orchestrator inputs — not vehicles detected in the image.
    """
    if not transcript_path.is_file():
        return 0, 0
    fusion_llm = REPO_ROOT / "fusion" / "llm" / "code"
    audio_code = REPO_ROOT / "perception" / "audio" / "code"
    for p in (fusion_llm, audio_code):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from llm_orchestrator import transcript_text  # noqa: E402
    from transcribe_run import (  # noqa: E402
        EMERGENCY_VEHICLE_PATTERN,
        NORMAL_VEHICLE_PATTERN,
        max_explicit_mentions,
    )

    data = json.loads(transcript_path.read_text(encoding="utf-8"))
    text = transcript_text(data)
    normal = max_explicit_mentions(text, NORMAL_VEHICLE_PATTERN)
    emergency = max_explicit_mentions(text, EMERGENCY_VEHICLE_PATTERN)
    or_more = _OR_MORE_VEHICLES_RE.search(text)
    if or_more:
        n = _parse_qty_token(or_more.group(1))
        if n is not None:
            normal = max(normal, n)
    return normal, emergency


def _median_int(values: List[int]) -> int:
    if not values:
        return 0
    return int(np.median(values))


def _match_column_metrics(rows: List[dict], match_key: str) -> Dict[str, Optional[float]]:
    eval_rows = [
        r
        for r in rows
        if r.get(match_key) is not None and not pd.isna(r.get(match_key))
    ]
    if not eval_rows:
        return {"n": 0, "accuracy": None}
    flags = [bool(r[match_key]) for r in eval_rows]
    return {"n": len(eval_rows), "accuracy": round(sum(flags) / len(flags), 4)}


def _exact_count_metrics(
    rows: List[dict],
    gt_key: str,
    pred_key: str,
) -> Dict[str, Optional[float]]:
    eval_rows = [
        r
        for r in rows
        if r.get(gt_key) is not None
        and not pd.isna(r.get(gt_key))
        and r.get(pred_key) is not None
        and not pd.isna(r.get(pred_key))
    ]
    if not eval_rows:
        return {"n": 0, "accuracy": None}
    correct_flags = [int(r[pred_key]) == int(r[gt_key]) for r in eval_rows]
    return {
        "n": len(eval_rows),
        "accuracy": round(sum(correct_flags) / len(correct_flags), 4),
    }


def _set_match_accuracy(rows: List[dict]) -> Dict[str, Optional[float]]:
    eval_rows = [
        r
        for r in rows
        if r.get("_gt_services_set") is not None and r.get("_pred_services_set") is not None
    ]
    if not eval_rows:
        return {"n": 0, "accuracy": None}
    exact = sum(
        1
        for r in eval_rows
        if set(r["_pred_services_set"]) == set(r["_gt_services_set"])
    )
    return {"n": len(eval_rows), "accuracy": round(exact / len(eval_rows), 4)}


def _collect_runs_by_audio(manifest_path: Path) -> Dict[str, List[dict]]:
    by_stem: Dict[str, List[dict]] = defaultdict(list)
    if not manifest_path.is_file():
        return by_stem
    with open(manifest_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            stem = str(row.get("audio_stem", "")).strip()
            rel = str(row.get("run_dir", "")).strip()
            if not stem or not rel:
                continue
            run_dir = REPO_ROOT / rel.replace("\\", "/")
            sls_path = run_dir / "fusion" / "sls.json"
            if not sls_path.is_file():
                continue
            by_stem[stem].append(
                {
                    "run_id": str(row.get("run_id", "")),
                    "sls_path": sls_path,
                    "transcript_path": run_dir / "perception" / "transcript.json",
                    "mono": bool(MONO_RUN_ID.match(str(row.get("run_id", "")))),
                }
            )
    return by_stem


def _aggregate_sls_mono(entries: List[dict]) -> Optional[dict]:
    mono = [e for e in entries if e["mono"]]
    if not mono:
        return None
    rep = mono[0]

    audio_persons: List[int] = []
    civs: List[int] = []
    resps: List[int] = []
    ff_roles: List[int] = []
    role_summaries: List[str] = []
    services: Set[str] = set()

    vn, ve = _radio_vehicle_counts(rep["transcript_path"])
    for entry in mono:
        sls = json.loads(entry["sls_path"].read_text(encoding="utf-8"))
        audio_persons.append(_sls_audio_person_count(sls))
        c, r, ff_n, roles = _sls_role_breakdown(sls)
        civs.append(c)
        resps.append(r)
        ff_roles.append(ff_n)
        if roles:
            role_summaries.append(roles)
        if not services:
            services = _services_from_sls(sls)

    return {
        "run_id": rep["run_id"],
        "n_mono_runs": len(mono),
        "pred_person_count": _median_int(audio_persons),
        "pred_civilian": _median_int(civs),
        "pred_firefighters": _median_int(resps),
        "pred_ff_roles": _median_int(ff_roles),
        "pred_roles": role_summaries[0] if role_summaries else "",
        "pred_vehicles_normal": vn,
        "pred_vehicles_emergency": ve,
        "pred_services": services,
    }


def _load_audio_cues_gt(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"audio_cues GT not found: {path}")
    df = pd.read_excel(path, sheet_name=0)
    return df.rename(columns=lambda c: str(c).strip())


def _match_flag(
    gt_val: object, pred_val: object, *, numeric: bool = True
) -> object:
    if gt_val is None or pred_val is None or pd.isna(gt_val) or pd.isna(pred_val):
        return pd.NA
    if numeric:
        return int(gt_val) == int(pred_val)
    return bool(gt_val) == bool(pred_val)


def _add_metric_row(
    rows: List[dict],
    audio_cue: str,
    stats: Dict[str, Optional[float]],
) -> None:
    rows.append(
        {
            "audio_cue": audio_cue,
            "n_evaluated": stats.get("n", 0),
            "accuracy": stats.get("accuracy"),
        }
    )


def evaluate_audio_cues(
    *,
    manifest_path: Path,
    gt_path: Path,
    report_path: Path,
) -> Path:
    gt_df = _load_audio_cues_gt(gt_path)
    runs_by_audio = _collect_runs_by_audio(manifest_path)
    ff_roles_col = "firefighter  roles"
    if ff_roles_col not in gt_df.columns:
        for col in gt_df.columns:
            if "firefighter" in col.lower() and "role" in col.lower():
                ff_roles_col = col
                break

    per_audio_rows: List[dict] = []

    for _, gt_row in gt_df.iterrows():
        audio_file = str(gt_row.get("audio", "")).strip()
        stem = _audio_stem(audio_file)
        gt_civilian = _parse_gt_count(gt_row.get("civilian mentioned"))
        gt_firefighters = _parse_gt_count(gt_row.get("firefighters mentioned"))
        gt_services = _parse_gt_services(gt_row.get("services needed"))
        gt_ff_roles_n, _ = _parse_gt_firefighter_roles(gt_row.get(ff_roles_col))
        entries = runs_by_audio.get(stem, [])
        mono_entries = [e for e in entries if e["mono"]]
        on_scene_n = (
            _load_on_scene_responders(mono_entries[0]["transcript_path"])
            if mono_entries
            else None
        )
        gt_person_count = _gt_person_count(
            gt_civilian, gt_firefighters, on_scene_n
        )
        gt_vehicles_text = str(gt_row.get("vehicles mentioned", "") or "").strip()
        gt_vehicles = _parse_gt_vehicles(gt_row.get("vehicles mentioned"))

        agg = _aggregate_sls_mono(entries)
        if agg is None:
            pred_person = pred_civilian = pred_ff = pred_ff_roles = None
            pred_roles = ""
            pred_v_normal = pred_v_emergency = None
            pred_services: Set[str] = set()
            run_id = ""
            n_mono = 0
        else:
            pred_person = agg["pred_person_count"]
            pred_civilian = agg["pred_civilian"]
            pred_ff = agg["pred_firefighters"]
            pred_ff_roles = agg["pred_ff_roles"]
            pred_roles = agg["pred_roles"]
            pred_v_normal = agg["pred_vehicles_normal"]
            pred_v_emergency = agg["pred_vehicles_emergency"]
            pred_services = agg["pred_services"]
            run_id = agg["run_id"]
            n_mono = agg["n_mono_runs"]

        m = re.match(r"^(scenario\d+)_audio(\d+)$", stem)
        scenario = m.group(1) if m else ""

        vehicles_match = pd.NA
        if pred_v_normal is not None:
            vehicles_match = _match_vehicles(
                gt_vehicles, int(pred_v_normal), int(pred_v_emergency)
            )

        gt_role_inference = _build_gt_role_inference(gt_civilian, gt_firefighters)
        gt_ff_role_inference = _build_gt_ff_role_inference(gt_ff_roles_n)
        role_inference_match = _match_roles(
            gt_role_inference, pred_civilian, pred_ff, pred_ff_roles
        )
        ff_role_inference_match = _match_roles(
            gt_ff_role_inference, pred_civilian, pred_ff, pred_ff_roles
        )

        pred_role_inference_text = (
            _format_pred_role_inference(int(pred_civilian), int(pred_ff))
            if pred_civilian is not None and pred_ff is not None
            else ""
        )
        pred_ff_role_text = (
            f"firefighter near fire:{int(pred_ff_roles)}"
            if pred_ff_roles is not None
            else ""
        )

        per_audio_rows.append(
            {
                "audio": audio_file,
                "scenario": scenario,
                "n_mono_runs": n_mono,
                "example_run_id": run_id,
                f"{METRIC_PEOPLE} (gt)": gt_person_count,
                f"{METRIC_PEOPLE} {PRED_SUFFIX}": pred_person,
                f"match {METRIC_PEOPLE}": _match_flag(gt_person_count, pred_person),
                f"{METRIC_VEHICLES} (gt)": gt_vehicles_text,
                f"{METRIC_VEHICLES} {PRED_SUFFIX}": (
                    f"normal:{pred_v_normal}, emergency:{pred_v_emergency}"
                    if pred_v_normal is not None
                    else ""
                ),
                f"match {METRIC_VEHICLES}": vehicles_match,
                f"{METRIC_ROLE_INFERENCE} (gt)": (
                    _format_roles(gt_role_inference) if gt_role_inference else ""
                ),
                f"{METRIC_ROLE_INFERENCE} {PRED_SUFFIX}": pred_role_inference_text,
                f"match {METRIC_ROLE_INFERENCE}": role_inference_match,
                f"{METRIC_FF_ROLE_INFERENCE} (gt)": (
                    _format_roles(gt_ff_role_inference) if gt_ff_role_inference else ""
                ),
                f"{METRIC_FF_ROLE_INFERENCE} {PRED_SUFFIX}": pred_ff_role_text,
                f"match {METRIC_FF_ROLE_INFERENCE}": ff_role_inference_match,
                f"{METRIC_SERVICES} (gt)": (
                    ",".join(sorted(gt_services)) if gt_services is not None else ""
                ),
                f"{METRIC_SERVICES} {PRED_SUFFIX}": ",".join(sorted(pred_services)),
                f"match {METRIC_SERVICES}": (
                    set(pred_services) == set(gt_services)
                    if gt_services is not None
                    else pd.NA
                ),
                f"inferred roles detail {PRED_SUFFIX}": pred_roles,
                "notes": str(gt_row.get("notes", "") or ""),
                "_gt_services_set": gt_services,
                "_pred_services_set": pred_services,
            }
        )

    metrics_rows: List[dict] = []
    _add_metric_row(
        metrics_rows,
        METRIC_PEOPLE,
        _exact_count_metrics(
            per_audio_rows,
            f"{METRIC_PEOPLE} (gt)",
            f"{METRIC_PEOPLE} {PRED_SUFFIX}",
        ),
    )
    _add_metric_row(
        metrics_rows,
        METRIC_VEHICLES,
        _match_column_metrics(per_audio_rows, f"match {METRIC_VEHICLES}"),
    )
    _add_metric_row(
        metrics_rows,
        METRIC_ROLE_INFERENCE,
        _match_column_metrics(per_audio_rows, f"match {METRIC_ROLE_INFERENCE}"),
    )
    _add_metric_row(
        metrics_rows,
        METRIC_FF_ROLE_INFERENCE,
        _match_column_metrics(per_audio_rows, f"match {METRIC_FF_ROLE_INFERENCE}"),
    )
    _add_metric_row(
        metrics_rows,
        METRIC_SERVICES,
        _set_match_accuracy(per_audio_rows),
    )

    per_audio_df = pd.DataFrame(per_audio_rows)
    drop_cols = [c for c in per_audio_df.columns if c.startswith("_")]
    per_audio_df = per_audio_df.drop(columns=drop_cols, errors="ignore")

    written = write_excel(
        report_path,
        {
            "metrics": pd.DataFrame(metrics_rows),
            "per_audio": per_audio_df,
        },
    )

    print(f"GT: {gt_path} ({len(per_audio_rows)} audio files)")
    print(f"Predictions: fusion/sls.json (runs 1 vista, mediana)")
    print(f"Wrote {written}")
    for row in metrics_rows:
        acc = row["accuracy"]
        acc_s = f"{acc:.2%}" if acc is not None and not pd.isna(acc) else "n/a"
        print(f"  {row['audio_cue']}: n={row['n_evaluated']}  accuracy={acc_s}")

    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate LLM/fusion SLS vs audio_cues.xlsx."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--gt", type=Path, default=AUDIO_CUES_XLSX)
    parser.add_argument("--output", type=Path, default=AUDIO_CUES_REPORT_XLSX)
    args = parser.parse_args()
    evaluate_audio_cues(
        manifest_path=args.manifest,
        gt_path=args.gt,
        report_path=args.output,
    )


if __name__ == "__main__":
    main()
