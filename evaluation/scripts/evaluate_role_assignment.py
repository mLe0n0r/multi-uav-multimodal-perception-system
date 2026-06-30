#!/usr/bin/env python3
"""
Role assignment evaluation against scene GT and audio-specific overrides.

Primary metric — role_assignment_accuracy (matched GT persons only):
  RoleAssignmentAccuracy = TP_role / N_matched
  where TP_role is GT persons that were detected/associated and have the correct role,
  and N_matched is GT persons with a known role that were successfully associated.
  Isolates role/LLM quality from detection — answers: "given a correct detection, was
  the role assigned correctly?"

Secondary metrics (sheet per_run / summary):
  - gt_role_recall: TP_role / N_gt (roles correct over all GT persons, includes misses)
  - exact_role_counts_match: whether total role counts in the SLS match expected totals

Ground truth: evaluation/groundTruth_data/visual/gt_roles.xlsx
  - per_scenario: roles + world positions per scenario (counts, audio exceptions)
  - per_img: img, id (visual.json), gt_role

Output: evaluation/results/sls_eval/role_assignment_metrics.xlsx

Usage:
  python evaluation/scripts/evaluate_role_assignment.py
  python evaluation/scripts/evaluate_role_assignment.py --run-id img74_aud2
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from eval_common import (  # noqa: E402
    AUDIO_CUES_XLSX,
    DEFAULT_MANIFEST,
    REPO_ROOT,
    ROLE_ASSIGNMENT_REPORT_XLSX,
    ROLE_MATCH_MAX_DIST_M,
    build_img_metadata,
    iter_mono_uav_runs,
    iter_multi_uav_runs,
    load_gt_roles_per_img,
    load_role_scenario_gt,
    write_excel,
    xy_from_position_field,
)

_FUSION_SLS = REPO_ROOT / "fusion" / "sls"
if str(_FUSION_SLS) not in sys.path:
    sys.path.insert(0, str(_FUSION_SLS))
from fused_counts import build_fused_object_list  # noqa: E402

RESPONDER_ROLES = frozenset({"firefighter", "possible_responder"})
PERSON_ROLES = ("civilian", "firefighter")
AUDIO_ROLE_COUNT_OVERRIDES: Dict[str, Dict[str, int]] = {
    # Speaker on radio + 2 firefighters mentioned on scene.
    "scenario3_audio3": {"civilian": 2, "firefighter": 3},
}
AUDIO_ONLY_EXPECTATIONS: Dict[str, Dict[str, int]] = {
    # No spatial person GT in scenario 5; casualties are occluded inside the fire truck.
    "scenario5_audio2": {"civilian": 2, "emergency_vehicle": 1},
}


def _norm_role(value: object) -> str:
    if value is None:
        return "unknown_person"
    text = str(value).strip().lower()
    if not text or text == "nan":
        return "unknown_person"
    if text in RESPONDER_ROLES:
        return "firefighter"
    if text == "civilian":
        return "civilian"
    return text


def _parse_cue_count(value: object) -> Optional[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip().lower()
    if text in ("---", "", "nan", "none"):
        return None
    if isinstance(value, (int, float)) and not pd.isna(value):
        return int(value)
    if text.isdigit():
        return int(text)
    return None


def _audio_stem(scenario_num: int, run_id: str) -> Optional[str]:
    match = re.search(r"aud(\d+)", run_id, re.IGNORECASE)
    if not match:
        return None
    return f"scenario{scenario_num}_audio{int(match.group(1))}"


def _load_audio_cues_by_stem() -> Dict[str, dict]:
    if not AUDIO_CUES_XLSX.is_file():
        raise FileNotFoundError(f"audio_cues GT not found: {AUDIO_CUES_XLSX}")
    df = pd.read_excel(AUDIO_CUES_XLSX, sheet_name=0)
    out: Dict[str, dict] = {}
    for _, row in df.iterrows():
        audio = str(row.get("audio", "")).strip()
        if not audio:
            continue
        stem = audio.replace(".wav", "")
        out[stem] = {str(k).strip(): row[k] for k in df.columns}
    return out


def _scene_role_counts(role_gt: Dict[int, List[dict]], scenario_num: int) -> Dict[str, int]:
    rows = role_gt.get(scenario_num, [])
    counts = {role: 0 for role in PERSON_ROLES}
    for row in rows:
        role = _norm_role(row.get("role"))
        if role in counts:
            counts[role] += 1
    return counts


def _expected_role_counts(
    *,
    scenario_num: int,
    audio_stem: Optional[str],
    role_gt: Dict[int, List[dict]],
    cue: Optional[dict],
) -> Tuple[Dict[str, int], str]:
    if audio_stem and audio_stem in AUDIO_ONLY_EXPECTATIONS:
        exp = {role: 0 for role in PERSON_ROLES}
        for role, n in AUDIO_ONLY_EXPECTATIONS[audio_stem].items():
            if role in exp:
                exp[role] = int(n)
        return exp, "audio_only_expectation"

    counts = _scene_role_counts(role_gt, scenario_num)
    source = "per_scenario" if counts else "audio_cues"

    if audio_stem and audio_stem in AUDIO_ROLE_COUNT_OVERRIDES:
        counts.update(AUDIO_ROLE_COUNT_OVERRIDES[audio_stem])
        source = "audio_override"

    if cue and not counts:
        civ = _parse_cue_count(cue.get("civilian mentioned"))
        ff = _parse_cue_count(cue.get("firefighters mentioned"))
        if civ is not None:
            counts["civilian"] = civ
        if ff is not None:
            counts["firefighter"] = ff

    return counts, source


def _expected_emergency_vehicle_count(
    *,
    audio_stem: Optional[str],
    cue: Optional[dict],
) -> Tuple[int, str]:
    if audio_stem and audio_stem in AUDIO_ONLY_EXPECTATIONS:
        n = int(AUDIO_ONLY_EXPECTATIONS[audio_stem].get("emergency_vehicle", 0))
        if n > 0:
            return n, "audio_only_expectation"
    if not cue:
        return 0, "none"
    vehicles = str(cue.get("vehicles mentioned", "")).lower()
    if "emergency vehicle" in vehicles or "fire truck" in vehicles or "firetruck" in vehicles:
        return 1, "audio_cues"
    return 0, "none"


def _sls_objects(sls_path: Path) -> List[dict]:
    data = json.loads(sls_path.read_text(encoding="utf-8"))
    return list(data.get("objects", []))


def _pred_role_counts(objects: List[dict]) -> Dict[str, int]:
    counts = {role: 0 for role in PERSON_ROLES}
    for obj in objects:
        if obj.get("class") != "person":
            continue
        role = _norm_role(obj.get("inferred_role"))
        if role in counts:
            counts[role] += 1
    return counts


def _pred_emergency_vehicle_count(objects: List[dict]) -> int:
    return sum(1 for obj in objects if obj.get("class") == "emergency_vehicle")


def _effective_gt_persons(
    *,
    scenario_num: int,
    audio_stem: Optional[str],
    role_gt: Dict[int, List[dict]],
) -> List[dict]:
    """
    GT person list for per-person role evaluation.

    Starts from gt_roles per_scenario rows; adds virtual audio-only GT slots when
    audio exceptions require people without a visual position (scenario5_audio2) or an
    extra role count (scenario3_audio3 third firefighter).
    """
    entries: List[dict] = []
    spatial = role_gt.get(scenario_num, [])
    for row in spatial:
        entries.append(
            {
                "role": _norm_role(row.get("role")),
                "x": float(row["x"]),
                "y": float(row["y"]),
                "match_mode": "spatial",
                "gt_source": "per_scenario",
            }
        )

    def _append_virtual(role: str, source: str, n: int) -> None:
        for _ in range(n):
            entries.append(
                {
                    "role": _norm_role(role),
                    "match_mode": "audio_only",
                    "gt_source": source,
                }
            )

    def _role_count(role: str) -> int:
        return sum(1 for entry in entries if entry["role"] == _norm_role(role))

    if audio_stem and audio_stem in AUDIO_ONLY_EXPECTATIONS:
        for role in PERSON_ROLES:
            need = int(AUDIO_ONLY_EXPECTATIONS[audio_stem].get(role, 0) or 0)
            gap = max(0, need - _role_count(role))
            _append_virtual(role, "audio_only_expectation", gap)

    if audio_stem and audio_stem in AUDIO_ROLE_COUNT_OVERRIDES:
        for role in PERSON_ROLES:
            need = int(AUDIO_ROLE_COUNT_OVERRIDES[audio_stem].get(role, 0) or 0)
            gap = max(0, need - _role_count(role))
            _append_virtual(role, "audio_override", gap)

    for idx, entry in enumerate(entries):
        entry["gt_idx"] = idx
    return entries


def _images_in_run(run_id: str) -> List[str]:
    return [f"img{int(n)}" for n in re.findall(r"img(\d+)", run_id, re.IGNORECASE)]


def _views_for_run(
    *,
    config: str,
    view_paths: List[Tuple[str, Path]],
    cross_view_path: Optional[Path],
) -> Tuple[List[dict], Optional[dict]]:
    if config == "multi" and cross_view_path and cross_view_path.is_file():
        cross_view = json.loads(cross_view_path.read_text(encoding="utf-8"))
        views = [
            {**copy, "_view_id": str(view_id)}
            for view_id, copy in (cross_view.get("visuals") or {}).items()
        ]
        return views, cross_view
    views: List[dict] = []
    for view_id, path in view_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        views.append({**payload, "_view_id": view_id})
    return views, None


def _pred_roles_by_view_and_visual_id(
    llm_path: Path,
    views: List[dict],
    matching: Optional[dict],
) -> Dict[Tuple[str, int], str]:
    if not llm_path.is_file():
        return {}
    llm = json.loads(llm_path.read_text(encoding="utf-8"))
    match_ctx = matching if matching and len(views) > 1 else None
    fused_objects, key_map = build_fused_object_list(views, match_ctx)

    fused_persons = [
        (int(obj["id"]), obj)
        for obj in fused_objects
        if obj.get("class") == "person" and obj.get("id") is not None
    ]
    roles_by_fused: Dict[int, str] = {}
    used_fused: set[int] = set()

    person_rows = [
        obj
        for obj in llm.get("objects", [])
        if obj.get("class") == "person" and not obj.get("audio_only")
    ]
    for obj in person_rows:
        role = _norm_role(obj.get("inferred_role"))
        if not role or role == "unknown_person":
            continue
        if obj.get("id") is not None:
            fid = int(obj["id"])
            if any(fid == f[0] for f in fused_persons):
                roles_by_fused[fid] = role
                used_fused.add(fid)

    for obj in person_rows:
        if obj.get("id") is not None and int(obj["id"]) in roles_by_fused:
            continue
        role = _norm_role(obj.get("inferred_role"))
        if not role or role == "unknown_person":
            continue
        spos = xy_from_position_field(obj.get("position"))
        if spos is None:
            continue
        best_fid: Optional[int] = None
        best_dist = float(ROLE_MATCH_MAX_DIST_M)
        for fid, fused in fused_persons:
            if fid in used_fused:
                continue
            fpos = xy_from_position_field(fused.get("position"))
            if fpos is None:
                continue
            dist = float(np.hypot(spos[0] - fpos[0], spos[1] - fpos[1]))
            if dist < best_dist:
                best_dist = dist
                best_fid = fid
        if best_fid is not None:
            roles_by_fused[best_fid] = role
            used_fused.add(best_fid)

    out: Dict[Tuple[str, int], str] = {}
    for (view_id, visual_id), fused_id in key_map.items():
        fused = fused_objects[fused_id] if fused_id < len(fused_objects) else None
        if not fused or fused.get("class") != "person":
            continue
        role = roles_by_fused.get(int(fused_id))
        if role:
            out[(str(view_id), int(visual_id))] = role
    return out


def _audio_only_persons_for_matching(objects: List[dict]) -> List[dict]:
    rows: List[dict] = []
    for idx, obj in enumerate(objects):
        if obj.get("class") != "person" or not obj.get("audio_only"):
            continue
        rows.append(
            {
                "person_idx": idx,
                "role": _norm_role(obj.get("inferred_role")),
            }
        )
    return rows


def _match_audio_only_gt(
    audio_gt: List[dict],
    audio_only_preds: List[dict],
) -> List[dict]:
    remaining = list(audio_only_preds)
    rows: List[dict] = []
    for gt in audio_gt:
        gt_role = _norm_role(gt.get("role"))
        base = {
            "gt_idx": gt["gt_idx"],
            "gt_role": gt_role,
            "gt_source": gt.get("gt_source", ""),
            "match_mode": "audio_only",
            "img": None,
            "visual_id": None,
            "pred_role": None,
            "role_correct": "no",
            "matched": "no",
        }
        for j, pred in enumerate(remaining):
            if pred["role"] == gt_role:
                pred = remaining.pop(j)
                rows.append(
                    {
                        **base,
                        "pred_person_idx": pred["person_idx"],
                        "pred_role": pred["role"],
                        "role_correct": "yes",
                        "matched": "yes",
                    }
                )
                break
        else:
            rows.append(base)
    return rows


def _detalhe_from_per_img(
    *,
    run_id: str,
    role_per_img: Dict[Tuple[str, int], str],
    pred_roles: Dict[Tuple[str, int], str],
) -> List[dict]:
    rows: List[dict] = []
    run_imgs = set(_images_in_run(run_id))
    gt_idx = 0
    for (img, visual_id), gt_role_raw in sorted(role_per_img.items()):
        if img not in run_imgs:
            continue
        gt_role = _norm_role(gt_role_raw)
        pred_role = pred_roles.get((img, visual_id))
        matched = pred_role is not None
        rows.append(
            {
                "gt_idx": gt_idx,
                "img": img,
                "visual_id": visual_id,
                "gt_role": gt_role,
                "gt_source": "per_img",
                "match_mode": "visual_id",
                "pred_role": pred_role,
                "role_correct": "yes" if matched and _norm_role(pred_role) == gt_role else "no",
                "matched": "yes" if matched else "no",
            }
        )
        gt_idx += 1
    return rows


def _roles_ok(expected: Dict[str, int], predicted: Dict[str, int]) -> bool:
    return all(predicted.get(role, 0) == expected.get(role, 0) for role in PERSON_ROLES)


def _evaluate_run(
    *,
    config: str,
    run_id: str,
    scenario_num: int,
    lighting: str,
    sls_path: Path,
    role_gt: Dict[int, List[dict]],
    role_per_img: Dict[Tuple[str, int], str],
    cues_by_stem: Dict[str, dict],
    views: List[dict],
    matching: Optional[dict],
) -> Tuple[dict, List[dict], List[dict]]:
    objects = _sls_objects(sls_path)
    stem = _audio_stem(scenario_num, run_id)
    cue = cues_by_stem.get(stem or "")

    expected, count_source = _expected_role_counts(
        scenario_num=scenario_num,
        audio_stem=stem,
        role_gt=role_gt,
        cue=cue,
    )
    predicted = _pred_role_counts(objects)

    exp_ev, ev_source = _expected_emergency_vehicle_count(audio_stem=stem, cue=cue)
    pred_ev = _pred_emergency_vehicle_count(objects)

    llm_path = sls_path.parent / "llm_output.json"
    pred_roles = _pred_roles_by_view_and_visual_id(llm_path, views, matching)
    detalhe = _detalhe_from_per_img(
        run_id=run_id,
        role_per_img=role_per_img,
        pred_roles=pred_roles,
    )

    if not detalhe:
        audio_gt = [
            g
            for g in _effective_gt_persons(
                scenario_num=scenario_num,
                audio_stem=stem,
                role_gt=role_gt,
            )
            if g.get("match_mode") == "audio_only"
        ]
        pred_audio_only = _audio_only_persons_for_matching(objects)
        detalhe = _match_audio_only_gt(audio_gt, pred_audio_only)

    n_visual_matched = sum(1 for r in detalhe if r.get("matched") == "yes")
    pred_audio_only = _audio_only_persons_for_matching(objects)
    for row in detalhe:
        row.update(
            {
                "run_id": run_id,
                "scenario": f"scenario{scenario_num}",
                "config": config,
                "lighting": lighting,
                "audio_stem": stem,
            }
        )

    contagem_rows: List[dict] = []
    for role in PERSON_ROLES:
        contagem_rows.append(
            {
                "run_id": run_id,
                "scenario": f"scenario{scenario_num}",
                "config": config,
                "lighting": lighting,
                "audio_stem": stem,
                "entity": role,
                "gt_count": int(expected.get(role, 0)),
                "pred_count": int(predicted.get(role, 0)),
                "count_error": int(predicted.get(role, 0)) - int(expected.get(role, 0)),
                "gt_source": count_source,
            }
        )
    if exp_ev > 0 or pred_ev > 0:
        contagem_rows.append(
            {
                "run_id": run_id,
                "scenario": f"scenario{scenario_num}",
                "config": config,
                "lighting": lighting,
                "audio_stem": stem,
                "entity": "emergency_vehicle",
                "gt_count": exp_ev,
                "pred_count": pred_ev,
                "count_error": pred_ev - exp_ev,
                "gt_source": ev_source,
            }
        )

    n_gt_persons = len(detalhe)
    n_matched = sum(1 for r in detalhe if r.get("matched") == "yes")
    n_role_correct = sum(
        1 for r in detalhe if r.get("matched") == "yes" and r.get("role_correct") == "yes"
    )
    role_assignment_accuracy = round(n_role_correct / n_matched, 4) if n_matched else None
    gt_role_recall = round(n_role_correct / n_gt_persons, 4) if n_gt_persons else None

    run_row = {
        "config": config,
        "run_id": run_id,
        "scenario": f"scenario{scenario_num}",
        "lighting": lighting,
        "audio_stem": stem,
        "n_gt_persons": n_gt_persons,
        "n_matched": n_matched,
        "n_role_correct": n_role_correct,
        "role_assignment_accuracy": role_assignment_accuracy,
        "gt_role_recall": gt_role_recall,
        "n_sls_persons": sum(1 for o in objects if o.get("class") == "person"),
        "n_visual_matched": n_visual_matched,
        "n_audio_only_persons": len(pred_audio_only),
        "exact_role_counts_match": bool(_roles_ok(expected, predicted)),
        "gt_civilian": expected.get("civilian", 0),
        "gt_firefighter": expected.get("firefighter", 0),
        "pred_civilian": predicted.get("civilian", 0),
        "pred_firefighter": predicted.get("firefighter", 0),
        "gt_emergency_vehicle": exp_ev,
        "pred_emergency_vehicle": pred_ev,
        "count_source": count_source,
    }
    return run_row, contagem_rows, detalhe


def _build_summary(per_run: pd.DataFrame, detalhe: pd.DataFrame) -> pd.DataFrame:
    if per_run.empty:
        return pd.DataFrame({"metric": ["no data"], "value": [""]})

    def _pool_role_assignment_accuracy() -> str:
        matched = detalhe[detalhe["matched"] == "yes"]
        if matched.empty:
            return ""
        return str(round(float((matched["role_correct"] == "yes").mean()), 4))

    def _pool_gt_role_recall() -> str:
        if detalhe.empty:
            return ""
        return str(round(float((detalhe["role_correct"] == "yes").mean()), 4))

    def _mean(col: str) -> str:
        if per_run.empty or col not in per_run.columns:
            return ""
        vals = per_run[col].dropna()
        if vals.empty:
            return ""
        return str(round(float(vals.mean()), 4))

    metrics = [
        ("role_assignment_accuracy (matched GT only)", "pool_matched"),
        ("gt_role_recall (all GT persons)", "pool_all_gt"),
        ("exact_role_counts_match (per run)", "exact_role_counts_match"),
        ("n_runs", "n_runs"),
    ]
    rows = []
    for label, col in metrics:
        if col == "pool_matched":
            rows.append((label, _pool_role_assignment_accuracy()))
        elif col == "pool_all_gt":
            rows.append((label, _pool_gt_role_recall()))
        elif col == "n_runs":
            rows.append((label, str(len(per_run))))
        else:
            rows.append((label, _mean(col)))
    return pd.DataFrame(rows, columns=["metric", "value"])


def evaluate_role_assignment(
    *,
    manifest_path: Path,
    report_path: Path,
    run_id_filter: Optional[str] = None,
) -> Path:
    role_gt = load_role_scenario_gt()
    role_per_img = load_gt_roles_per_img()
    cues_by_stem = _load_audio_cues_by_stem()
    img_lighting = build_img_metadata()
    run_rows: List[dict] = []
    contagem_rows: List[dict] = []
    detalhe_rows: List[dict] = []

    def _lighting(run_id: str, img_ref: int) -> str:
        for n, light in img_lighting.items():
            if f"img{n}" in run_id:
                return light
        return img_lighting.get(img_ref, "unknown")

    def _process(
        config: str,
        run_id: str,
        scenario_num: int,
        img_ref: int,
        sls_path: Optional[Path],
        view_paths: List[Tuple[str, Path]],
        cross_view_path: Optional[Path] = None,
    ) -> None:
        if not sls_path or not sls_path.is_file():
            return
        cv_path = cross_view_path
        if config == "multi" and (cv_path is None or not cv_path.is_file()):
            cv_path = sls_path.parent / "cross_view.json"
        views, matching = _views_for_run(
            config=config,
            view_paths=view_paths,
            cross_view_path=cv_path if config == "multi" else None,
        )
        run_row, contagem, detalhe = _evaluate_run(
            config=config,
            run_id=run_id,
            scenario_num=scenario_num,
            lighting=_lighting(run_id, img_ref),
            sls_path=sls_path,
            role_gt=role_gt,
            role_per_img=role_per_img,
            cues_by_stem=cues_by_stem,
            views=views,
            matching=matching,
        )
        run_rows.append(run_row)
        contagem_rows.extend(contagem)
        detalhe_rows.extend(detalhe)

    for run in iter_mono_uav_runs(manifest_path):
        if run_id_filter and run.run_id != run_id_filter:
            continue
        _process(
            "mono",
            run.run_id,
            run.scenario_num,
            run.img_ref,
            run.sls_json,
            [(run.view_id, run.visual_json)],
        )

    for run in iter_multi_uav_runs(manifest_path):
        if run_id_filter and run.run_id != run_id_filter:
            continue
        _process(
            "multi",
            run.run_id,
            run.scenario_num,
            run.img_ref,
            run.sls_json,
            [(run.view_ref_id, run.visual_ref), (run.view_other_id, run.visual_other)],
            cross_view_path=run.cross_view_json,
        )

    per_run = pd.DataFrame(run_rows)
    contagem = pd.DataFrame(contagem_rows)
    detalhe = pd.DataFrame(detalhe_rows)
    summary = _build_summary(per_run, detalhe)
    path = write_excel(
        report_path,
        {
            "summary": summary,
            "per_run": per_run,
            "contagem_roles": contagem,
            "detalhe": detalhe,
        },
    )
    print(f"Wrote {path} ({len(per_run)} runs)")
    print(summary.to_string(index=False))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Role assignment vs scene GT + audio overrides")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=ROLE_ASSIGNMENT_REPORT_XLSX)
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()
    evaluate_role_assignment(
        manifest_path=args.manifest,
        report_path=args.output,
        run_id_filter=args.run_id,
    )


if __name__ == "__main__":
    main()
