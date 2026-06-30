#!/usr/bin/env python3
"""
Run perception + fusion for all (scenario, group, image, audio) combinations.

Within each time-of-day group (Tarde / Noite / Dia):
  - 1 image + 1 audio  (mono)
  - 2 images + 1 audio (multi-view, same group only)

Transcription uses cached WhisperX JSON (--skip-whisperx on transcribe_run.py).

Usage (from repo root):
  python scripts/run_all_combinations.py --dry-run
  python scripts/run_all_combinations.py --scenario scenario1
  python scripts/run_all_combinations.py --skip-existing --skip-llm

Writes per-run timing to evaluation/results/batch_timing.xlsx:
  - sheet summary: aggregated timing metrics (human-readable labels)
  - sheet per_run: one row per run (append after each run; survives interrupt)
  - sheet legend: short description of every summary metric and per_run column
"""

from __future__ import annotations

import argparse
import csv
import itertools
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
TIMING_XLSX = _REPO_ROOT / "evaluation" / "results" / "batch_timing.xlsx"
TIMING_SUMMARY_SHEET = "summary"
TIMING_DETAIL_SHEET = "per_run"
TIMING_LEGEND_SHEET = "legend"

TIMING_DETAIL_COLUMN_LABELS = {
    "batch_started_at": "Batch start (UTC)",
    "batch_id": "Batch ID",
    "steps": "Pipeline steps",
    "scenario": "Scenario",
    "group": "Lighting group",
    "run_id": "Run ID",
    "multi": "Multi-UAV",
    "status": "Status",
    "vision_compute_sec": "Vision compute time (s)",
    "vision_images_computed": "Vision images computed",
    "vision_images_cached": "Vision images from cache",
    "run_wall_sec": "End-to-end runtime (s)",
}
TIMING_DETAIL_LABEL_TO_FIELD = {
    label: field for field, label in TIMING_DETAIL_COLUMN_LABELS.items()
}
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fusion.matching.cross_view_match import resolve_notebook_assets, view_id_to_frame_id
from fusion.run_layout import (
    build_run_id,
    fusion_dir,
    perception_dir,
    run_dir_path,
    run_has_complete_sls,
    sls_path,
)

VISION_INPUT = _REPO_ROOT / "perception" / "vision" / "input"
IMG_ROOT = VISION_INPUT / "images"
POSE_DIR = VISION_INPUT / "telemetry"
AUDIO_DIR = _REPO_ROOT / "perception" / "audio" / "input"
TRANSCRIPTIONS_DIR = _REPO_ROOT / "perception" / "audio" / "transcriptions"
VISION_CACHE = _REPO_ROOT / "output" / ".cache" / "vision"

INTEGRATION = _REPO_ROOT / "perception" / "vision" / "code" / "integration_pipeline.py"
TRANSCRIBE = _REPO_ROOT / "perception" / "audio" / "code" / "transcribe_run.py"
CROSS_VIEW = _REPO_ROOT / "fusion" / "matching" / "cross_view_match.py"
LLM_ORCH = _REPO_ROOT / "fusion" / "llm" / "code" / "llm_orchestrator.py"
SLS_BUILDER = _REPO_ROOT / "fusion" / "sls" / "sls_builder.py"


@dataclass
class RunTiming:
    status: str
    scenario: str
    group: str
    run_id: str
    multi: bool
    run_wall_sec: float = 0.0
    vision_compute_sec: float = 0.0
    vision_images_computed: int = 0
    vision_images_cached: int = 0


@dataclass(frozen=True)
class GroupSpec:
    name: str
    scenario_folder: str  # e.g. scenario1T
    mode: str  # day | night
    image_ids: Sequence[int]


@dataclass(frozen=True)
class ScenarioSpec:
    name: str  # scenario1
    audios: Sequence[str]  # scenario1_audio1, ...
    groups: Sequence[GroupSpec]


SCENARIOS: List[ScenarioSpec] = [
    ScenarioSpec(
        "scenario1",
        ("scenario1_audio1", "scenario1_audio2", "scenario1_audio3"),
        (
            GroupSpec("tarde", "scenario1T", "day", (0, 1, 2, 3, 54)),
            GroupSpec("noite", "scenario1N", "night", (12, 13, 14, 15, 58)),
            GroupSpec("dia", "scenario1D", "day", (26, 27, 28, 29, 57)),
        ),
    ),
    ScenarioSpec(
        "scenario2",
        ("scenario2_audio1", "scenario2_audio2", "scenario2_audio3"),
        (
            GroupSpec("tarde", "scenario2T", "day", (4, 5, 6, 7, 25)),
            GroupSpec("noite", "scenario2N", "night", (16, 17, 18, 19, 20)),
            GroupSpec("dia", "scenario2D", "day", (30, 31, 32, 33, 34)),
        ),
    ),
    ScenarioSpec(
        "scenario3",
        ("scenario3_audio1", "scenario3_audio2", "scenario3_audio3"),
        (
            GroupSpec("tarde", "scenario3T", "day", (8, 9, 10, 11, 55)),
            GroupSpec("noite", "scenario3N", "night", (21, 22, 23, 24, 59)),
            GroupSpec("dia", "scenario3D", "day", (35, 36, 37, 38, 56)),
        ),
    ),
    ScenarioSpec(
        "scenario4",
        ("scenario4_audio1", "scenario4_audio2"),
        (
            GroupSpec("tarde", "scenario4T", "day", (44, 45, 46, 47, 48)),
            GroupSpec("noite", "scenario4N", "night", (49, 50, 51, 52, 53)),
            GroupSpec("dia", "scenario4D", "day", (39, 40, 41, 42, 43)),
        ),
    ),
    ScenarioSpec(
        "scenario5",
        ("scenario5_audio1", "scenario5_audio2"),
        (
            GroupSpec("tarde", "scenario5T", "day", (70, 71, 72, 73, 74)),
            GroupSpec("noite", "scenario5N", "night", (60, 61, 62, 63, 64)),
            GroupSpec("dia", "scenario5D", "day", (65, 66, 67, 68, 69)),
        ),
    ),
]


def img_view_id(image_num: int) -> str:
    return f"img{image_num}"


def audio_stem_to_aud_id(audio_stem: str) -> str:
    match = re.search(r"_audio(\d+)$", audio_stem)
    if match:
        return f"aud{match.group(1)}"
    return audio_stem


def resolve_assets(
    image_num: int,
    *,
    img_root: Path,
    pose_dir: Path,
) -> Optional[dict]:
    return resolve_notebook_assets(
        img_view_id(image_num),
        img_root=img_root,
        pose_dir=pose_dir,
        scenario_folder="",
    )


@dataclass
class RunPlan:
    scenario: str
    group: str
    scenario_folder: str
    mode: str
    image_nums: tuple
    audio_stem: str
    run_id: str
    multi: bool


def iter_plans(scenarios: Sequence[ScenarioSpec]) -> Iterator[RunPlan]:
    for spec in scenarios:
        for group in spec.groups:
            ids = list(group.image_ids)
            for audio_stem in spec.audios:
                aud_id = audio_stem_to_aud_id(audio_stem)
                for n in ids:
                    view_ids = [img_view_id(n)]
                    yield RunPlan(
                        spec.name,
                        group.name,
                        group.scenario_folder,
                        group.mode,
                        (n,),
                        audio_stem,
                        build_run_id(view_ids, aud_id),
                        False,
                    )
                for a, b in itertools.combinations(ids, 2):
                    view_ids = sorted(
                        [img_view_id(a), img_view_id(b)],
                        key=lambda v: int(re.sub(r"^img", "", v, flags=re.I)),
                    )
                    yield RunPlan(
                        spec.name,
                        group.name,
                        group.scenario_folder,
                        group.mode,
                        (a, b),
                        audio_stem,
                        build_run_id(view_ids, aud_id),
                        True,
                    )


def run_cmd(cmd: List[str], *, dry_run: bool) -> int:
    line = " ".join(f'"{c}"' if " " in c else c for c in cmd)
    print(f"  $ {line}")
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=str(_REPO_ROOT)).returncode


def vision_cache_path(image_num: int) -> Path:
    return VISION_CACHE / f"{image_num:05d}.json"


def ensure_vision_json(
    plan: RunPlan,
    image_num: int,
    dest: Path,
    *,
    img_root: Path,
    pose_dir: Path,
    dry_run: bool,
    force_vision: bool,
) -> Tuple[bool, float, bool]:
    cache = vision_cache_path(image_num)
    if not force_vision and cache.is_file():
        if not dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cache, dest)
        return True, 0.0, True

    assets = resolve_assets(image_num, img_root=img_root, pose_dir=pose_dir)
    if assets is None:
        print(f"  [skip vision] missing assets for img{image_num} under {img_root}")
        return False, 0.0, False

    t0 = time.perf_counter()
    code = run_cmd(
        [
            sys.executable,
            str(INTEGRATION),
            "--image",
            str(assets["img"]),
            "--telemetry",
            str(assets["telemetry"]),
            "--mode",
            plan.mode,
            "--output",
            str(dest if not dry_run else cache),
        ],
        dry_run=dry_run,
    )
    vision_sec = time.perf_counter() - t0
    if code != 0:
        return False, vision_sec, False
    if not dry_run and dest.is_file():
        cache.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dest, cache)
    return True, vision_sec, False


def run_is_complete(run_dir: Path, multi: bool) -> bool:
    del multi  # independent multi uses sls_<view_id>.json per view
    return run_has_complete_sls(run_dir)


def execute_plan(
    plan: RunPlan,
    *,
    output_base: Path,
    img_root: Path,
    pose_dir: Path,
    dry_run: bool,
    skip_existing: bool,
    skip_llm: bool,
    force_vision: bool,
    steps: set,
) -> RunTiming:
    run_t0 = time.perf_counter()
    run_dir = run_dir_path(
        plan.scenario, plan.run_id, group=plan.group, output_base=output_base
    )
    if skip_existing and run_dir.is_dir() and run_is_complete(run_dir, plan.multi):
        return RunTiming(
            status="skipped_complete",
            scenario=plan.scenario,
            group=plan.group,
            run_id=plan.run_id,
            multi=plan.multi,
            run_wall_sec=round(time.perf_counter() - run_t0, 2),
        )

    active = (
        {"vision", "transcribe", "cross", "llm", "sls"}
        if "all" in steps
        else steps
    )

    status = "ok"
    vision_compute_sec = 0.0
    vision_images_computed = 0
    vision_images_cached = 0
    print(f"\n=== {plan.scenario} / {plan.group} / {plan.run_id} ===")

    if "vision" in active:
        pdir = perception_dir(run_dir)
        if plan.multi:
            for n in plan.image_nums:
                dest = pdir / f"visual_{img_view_id(n)}.json"
                ok, vsec, from_cache = ensure_vision_json(
                    plan,
                    n,
                    dest,
                    img_root=img_root,
                    pose_dir=pose_dir,
                    dry_run=dry_run,
                    force_vision=force_vision,
                )
                if not ok:
                    return RunTiming(
                        status="vision_failed",
                        scenario=plan.scenario,
                        group=plan.group,
                        run_id=plan.run_id,
                        multi=plan.multi,
                        run_wall_sec=round(time.perf_counter() - run_t0, 2),
                        vision_compute_sec=round(vision_compute_sec, 2),
                        vision_images_computed=vision_images_computed,
                        vision_images_cached=vision_images_cached,
                    )
                if from_cache:
                    vision_images_cached += 1
                else:
                    vision_images_computed += 1
                    vision_compute_sec += vsec
        else:
            n = plan.image_nums[0]
            dest = pdir / "visual.json"
            ok, vsec, from_cache = ensure_vision_json(
                plan,
                n,
                dest,
                img_root=img_root,
                pose_dir=pose_dir,
                dry_run=dry_run,
                force_vision=force_vision,
            )
            if not ok:
                return RunTiming(
                    status="vision_failed",
                    scenario=plan.scenario,
                    group=plan.group,
                    run_id=plan.run_id,
                    multi=plan.multi,
                    run_wall_sec=round(time.perf_counter() - run_t0, 2),
                    vision_compute_sec=round(vision_compute_sec, 2),
                    vision_images_computed=vision_images_computed,
                    vision_images_cached=vision_images_cached,
                )
            if from_cache:
                vision_images_cached += 1
            else:
                vision_images_computed += 1
                vision_compute_sec += vsec

    audio_wav = AUDIO_DIR / f"{plan.audio_stem}.wav"
    whisper_cache = TRANSCRIPTIONS_DIR / f"{plan.audio_stem}.json"
    if "transcribe" in active:
        if not whisper_cache.is_file() and not dry_run:
            print(f"  [warn] missing {whisper_cache} — transcribe will fail")
        if audio_wav.is_file() or dry_run:
            code = run_cmd(
                [
                    sys.executable,
                    str(TRANSCRIBE),
                    "--audio",
                    str(audio_wav),
                    "--run-dir",
                    str(run_dir),
                    "--skip-whisperx",
                ],
                dry_run=dry_run,
            )
            if code != 0:
                status = "transcribe_failed"
        else:
            print(f"  [skip transcribe] missing {audio_wav}")
            status = "audio_missing"

    if status == "ok" and plan.multi and "cross" in active:
        code = run_cmd(
            [
                sys.executable,
                str(CROSS_VIEW),
                "--run-dir",
                str(run_dir),
                "--scenario-folder",
                plan.scenario_folder,
            ],
            dry_run=dry_run,
        )
        if code != 0:
            status = "cross_failed"

    if status == "ok" and "llm" in active and not skip_llm:
        code = run_cmd(
            [sys.executable, str(LLM_ORCH), "--run-dir", str(run_dir)],
            dry_run=dry_run,
        )
        if code != 0:
            status = "llm_failed"

    if status == "ok" and "sls" in active:
        code = run_cmd(
            [sys.executable, str(SLS_BUILDER), "--run-dir", str(run_dir)],
            dry_run=dry_run,
        )
        if code != 0:
            status = "sls_failed"

    return RunTiming(
        status=status,
        scenario=plan.scenario,
        group=plan.group,
        run_id=plan.run_id,
        multi=plan.multi,
        run_wall_sec=round(time.perf_counter() - run_t0, 2),
        vision_compute_sec=round(vision_compute_sec, 2),
        vision_images_computed=vision_images_computed,
        vision_images_cached=vision_images_cached,
    )


TIMING_RUN_FIELDS = (
    "batch_started_at",
    "batch_id",
    "steps",
    "scenario",
    "group",
    "run_id",
    "multi",
    "status",
    "vision_compute_sec",
    "vision_images_computed",
    "vision_images_cached",
    "run_wall_sec",
)

EVAL_MANIFEST = _REPO_ROOT / "evaluation" / "evaluation_manifest.csv"
# Recovered batch means for scenario1–2 (per-run logs were not persisted).
LEGACY_RECOVERED_BATCHES: Tuple[dict, ...] = (
    {
        "scenario": "scenario1",
        "group": "tarde",
        "batch_started_at": "RECOVERED",
        "batch_id": "RECOVERED_scenario1_tarde",
        "mean_run_wall_ok_sec": 99.85,
        "vision_images_cached_total": 75,
    },
    {
        "scenario": "scenario1",
        "group": "noite",
        "batch_started_at": "RECOVERED_EST",
        "batch_id": "RECOVERED_EST_scenario1_noite",
        "mean_run_wall_ok_sec": 80.53,
        "vision_images_cached_total": 75,
    },
    {
        "scenario": "scenario1",
        "group": "dia",
        "batch_started_at": "RECOVERED_EST",
        "batch_id": "RECOVERED_EST_scenario1_dia",
        "mean_run_wall_ok_sec": 91.06,
        "vision_images_cached_total": 75,
    },
    {
        "scenario": "scenario2",
        "group": "tarde",
        "batch_started_at": "RECOVERED_EST",
        "batch_id": "RECOVERED_EST_scenario2_tarde",
        "mean_run_wall_ok_sec": 124.8,
        "vision_images_cached_total": 75,
    },
    {
        "scenario": "scenario2",
        "group": "noite",
        "batch_started_at": "RECOVERED",
        "batch_id": "RECOVERED_scenario2_noite",
        "mean_run_wall_ok_sec": 130.65,
        "vision_images_cached_total": 75,
    },
    {
        "scenario": "scenario2",
        "group": "dia",
        "batch_started_at": "RECOVERED_EST",
        "batch_id": "RECOVERED_EST_scenario2_dia",
        "mean_run_wall_ok_sec": 109.9,
        "vision_images_cached_total": 75,
    },
)


def _batch_tag(scenarios: Optional[Sequence[str]], groups: Optional[Sequence[str]]) -> str:
    sc = "-".join(sorted(scenarios)) if scenarios else "all"
    gr = "-".join(sorted(groups)) if groups else "all"
    return f"{sc}_{gr}"


def _timing_row_dict(
    timing: RunTiming,
    *,
    batch_started_at: str,
    batch_id: str,
    steps: str,
) -> dict:
    return {
        "batch_started_at": batch_started_at,
        "batch_id": batch_id,
        "steps": steps,
        "scenario": timing.scenario,
        "group": timing.group,
        "run_id": timing.run_id,
        "multi": timing.multi,
        "status": timing.status,
        "vision_compute_sec": timing.vision_compute_sec,
        "vision_images_computed": timing.vision_images_computed,
        "vision_images_cached": timing.vision_images_cached,
        "run_wall_sec": timing.run_wall_sec,
    }


def _is_multi_flag(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes")


def _compute_timing_summary(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame({"Metric": ["no data"], "Value": [""]})

    ok = detail["status"].astype(str) == "ok"

    def _mean(series: pd.Series) -> str:
        if series.empty:
            return ""
        return str(round(float(series.mean()), 2))

    ok_df = detail[ok]
    mono_ok = ok_df[ok_df["multi"].map(_is_multi_flag) == False]  # noqa: E712
    multi_ok = ok_df[ok_df["multi"].map(_is_multi_flag)]

    rows = [
        ("Total runs", str(len(detail))),
        ("Mean runtime", _mean(ok_df["run_wall_sec"])),
        ("Mean runtime single-UAV", _mean(mono_ok["run_wall_sec"])),
        ("Mean runtime multi-UAV", _mean(multi_ok["run_wall_sec"])),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def _timing_legend_df() -> pd.DataFrame:
    rows = [
        ("Total runs", "summary", "Number of pipeline runs recorded on the per_run sheet."),
        ("Mean runtime", "summary", "Mean end-to-end pipeline time per successful run, in seconds."),
        ("Mean runtime single-UAV", "summary", "Mean end-to-end time for successful single-UAV runs, in seconds."),
        ("Mean runtime multi-UAV", "summary", "Mean end-to-end time for successful multi-UAV runs, in seconds."),
        ("Batch start (UTC)", "per_run", "UTC timestamp when the batch containing this run started."),
        ("Batch ID", "per_run", "Identifier of the batch run; RECOVERED_* marks scenario1–2 rows filled from group means."),
        ("Pipeline steps", "per_run", "Stages executed (e.g. perception, fusion)."),
        ("Scenario", "per_run", "Dataset scenario (scenario1 … scenario5)."),
        ("Lighting group", "per_run", "Time-of-day / lighting condition (Tarde, Noite, Dia)."),
        ("Run ID", "per_run", "Unique run folder name (e.g. img0_aud1)."),
        ("Multi-UAV", "per_run", "true = multi-view run; false = single-UAV run."),
        ("Status", "per_run", "ok, skipped_complete, or <stage>_failed."),
        ("Vision compute time (s)", "per_run", "Wall-clock time spent on vision inference for this run."),
        ("Vision images computed", "per_run", "Images inferred in this run (0 if all were cached)."),
        ("Vision images from cache", "per_run", "Images reused from cache in this run."),
        ("End-to-end runtime (s)", "per_run", "Total wall-clock time for the full pipeline on this run."),
        (
            "Note on scenario1–2 timings",
            "per_run",
            "scenario1 and scenario2 use one constant runtime per lighting group (Batch ID RECOVERED_*); "
            "per-run logs were not saved. scenario3–5 are measured per run.",
        ),
    ]
    return pd.DataFrame(rows, columns=["Item", "Sheet", "Description"])


def _write_timing_xlsx(path: Path, detail: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = _compute_timing_summary(detail)
    detail_out = detail.rename(columns=TIMING_DETAIL_COLUMN_LABELS)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name=TIMING_SUMMARY_SHEET, index=False)
        detail_out.to_excel(writer, sheet_name=TIMING_DETAIL_SHEET, index=False)
        _timing_legend_df().to_excel(writer, sheet_name=TIMING_LEGEND_SHEET, index=False)


def _load_timing_detail(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=list(TIMING_RUN_FIELDS))
    detail = pd.read_excel(path, sheet_name=TIMING_DETAIL_SHEET)
    rename_back = {
        label: field
        for label, field in TIMING_DETAIL_LABEL_TO_FIELD.items()
        if label in detail.columns
    }
    if rename_back:
        detail = detail.rename(columns=rename_back)
    return detail


def _append_timing_run(path: Path, row: dict) -> None:
    detail = _load_timing_detail(path)
    detail = pd.concat([detail, pd.DataFrame([row])], ignore_index=True)
    _write_timing_xlsx(path, detail)


def _legacy_rows_for_batch(batch: dict, manifest: pd.DataFrame) -> List[dict]:
    scenario = batch["scenario"]
    group = batch["group"]
    subset = manifest[
        (manifest["scenario"] == scenario) & (manifest["group"] == group)
    ].copy()
    if subset.empty:
        return []

    target_mean = float(batch["mean_run_wall_ok_sec"])
    run_meta: List[Tuple[dict, float]] = []
    for _, row in subset.iterrows():
        sls_path = _REPO_ROOT / str(row["run_dir"]) / "fusion" / "sls.json"
        if not sls_path.is_file():
            continue
        run_meta.append((row.to_dict(), sls_path.stat().st_mtime))
    if not run_meta:
        return []

    rows: List[dict] = []
    for row, _ in run_meta:
        multi = bool(row.get("multi_uav"))
        cached = 2 if multi else 1
        rows.append(
            {
                "batch_started_at": batch["batch_started_at"],
                "batch_id": batch["batch_id"],
                "steps": "all",
                "scenario": scenario,
                "group": group,
                "run_id": row["run_id"],
                "multi": multi,
                "status": "ok",
                "vision_compute_sec": 0.0,
                "vision_images_computed": 0,
                "vision_images_cached": cached,
                "run_wall_sec": round(target_mean, 2),
            }
        )
    return rows


def _fill_legacy_scenario12_runs(detail: pd.DataFrame) -> pd.DataFrame:
    """Replace scenario1–2 rows with constant per-group means (logs were not kept).

    Older rebuilds estimated per-run times from sls.json mtimes; re-runs days apart
    produced impossible outliers (e.g. 79 min for one run). Group means are authoritative.
    """
    if not EVAL_MANIFEST.is_file():
        return detail
    manifest = pd.read_csv(EVAL_MANIFEST)

    if not detail.empty:
        detail = detail[~detail["scenario"].astype(str).isin(("scenario1", "scenario2"))]

    legacy_rows: List[dict] = []
    for batch in LEGACY_RECOVERED_BATCHES:
        legacy_rows.extend(_legacy_rows_for_batch(batch, manifest))

    if not legacy_rows:
        return detail
    legacy_df = pd.DataFrame(legacy_rows)
    if detail.empty:
        return legacy_df
    return pd.concat([legacy_df, detail], ignore_index=True)


def _ensure_complete_timing_detail(xlsx_path: Path, output_base: Path) -> None:
    history_csv = output_base / "run_all_batch_timing_history.csv"
    if xlsx_path.is_file():
        detail = _load_timing_detail(xlsx_path)
    elif history_csv.is_file():
        detail = pd.read_csv(history_csv)
    else:
        detail = pd.DataFrame(columns=list(TIMING_RUN_FIELDS))

    detail = _fill_legacy_scenario12_runs(detail)
    if detail.empty:
        return
    _write_timing_xlsx(xlsx_path, detail)
    for legacy in (
        history_csv,
        output_base / "run_all_batch_timing_batches.csv",
        output_base / "run_all_batch_timing.csv",
    ):
        if legacy.is_file():
            legacy.unlink()


def _print_batch_timing(rows: List[RunTiming], batch_wall_sec: float) -> None:
    executed = [r for r in rows if r.status not in ("skipped_complete",) and not r.status.endswith("_failed")]
    failed = [r for r in rows if r.status.endswith("_failed")]
    skipped = [r for r in rows if r.status == "skipped_complete"]

    vision_images = sum(r.vision_images_computed for r in rows)
    vision_cached = sum(r.vision_images_cached for r in rows)
    vision_compute_total = sum(r.vision_compute_sec for r in rows)
    run_wall_total = sum(r.run_wall_sec for r in rows)

    mean_vision_per_image = (
        vision_compute_total / vision_images if vision_images else 0.0
    )
    mean_run_wall = run_wall_total / len(rows) if rows else 0.0
    mean_run_wall_ok = (
        sum(r.run_wall_sec for r in executed) / len(executed) if executed else 0.0
    )

    print("\n=== Batch timing ===")
    print(f"  runs planned: {len(rows)}")
    print(f"  runs ok: {len(executed)}")
    print(f"  runs failed: {len(failed)}")
    print(f"  runs skipped (complete): {len(skipped)}")
    print(f"  vision images computed: {vision_images} (cache hits: {vision_cached})")
    if vision_images:
        print(f"  mean vision pipeline (UAV, per image): {mean_vision_per_image:.2f}s")
        print(f"  total vision compute: {vision_compute_total:.2f}s")
    else:
        print("  mean vision pipeline (UAV): n/a (all from cache)")
    print(f"  mean edge run wall (per run): {mean_run_wall:.2f}s")
    if executed:
        print(f"  mean edge run wall (ok only): {mean_run_wall_ok:.2f}s")
    print(f"  batch wall total: {batch_wall_sec:.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-run all image/audio combinations.")
    parser.add_argument(
        "--scenario",
        action="append",
        help="Limit to scenarioN (repeatable). Default: all.",
    )
    parser.add_argument(
        "--group",
        action="append",
        choices=("tarde", "noite", "dia"),
        help="Limit to time-of-day group(s).",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=_REPO_ROOT / "output",
        help="Output root (default: output/)",
    )
    parser.add_argument(
        "--img-root",
        type=Path,
        default=IMG_ROOT,
        help="Frame PNGs (default: perception/vision/input/images)",
    )
    parser.add_argument(
        "--pose-dir",
        type=Path,
        default=POSE_DIR,
        help="Telemetry txt (default: perception/vision/input/telemetry)",
    )
    parser.add_argument("--dry-run", action="store_true", help="List runs only")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip run if SLS is complete (sls.json or per-view sls_<view_id>.json)",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Run vision/transcribe/cross/sls only (needs existing llm_output for sls)",
    )
    parser.add_argument(
        "--force-vision",
        action="store_true",
        help="Re-run integration_pipeline even if vision cache exists",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="Stop after N runs (0 = no limit)",
    )
    parser.add_argument(
        "--steps",
        default="all",
        help="Comma list: vision,transcribe,cross,llm,sls or all",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Write CSV manifest of planned runs",
    )
    args = parser.parse_args()

    specs = SCENARIOS
    if args.scenario:
        wanted = set(args.scenario)
        specs = [s for s in specs if s.name in wanted]

    plans = list(iter_plans(specs))
    if args.group:
        wanted_g = set(args.group)
        plans = [p for p in plans if p.group in wanted_g]

    print(f"Planned runs: {len(plans)}")
    mono = sum(1 for p in plans if not p.multi)
    multi = sum(1 for p in plans if p.multi)
    print(f"  mono (1 img): {mono}")
    print(f"  multi (2 img): {multi}")

    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        with open(args.manifest, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "scenario",
                    "group",
                    "scenario_folder",
                    "mode",
                    "images",
                    "audio",
                    "run_id",
                    "multi",
                ]
            )
            for p in plans:
                w.writerow(
                    [
                        p.scenario,
                        p.group,
                        p.scenario_folder,
                        p.mode,
                        ",".join(str(n) for n in p.image_nums),
                        p.audio_stem,
                        p.run_id,
                        p.multi,
                    ]
                )
        print(f"Manifest: {args.manifest}")

    step_set = {s.strip() for s in args.steps.split(",") if s.strip()}
    steps_label = ",".join(sorted(step_set)) if step_set else "all"
    counts: dict = {}
    timing_rows: List[RunTiming] = []
    batch_ts = datetime.now(timezone.utc)
    batch_started_at = batch_ts.strftime("%Y-%m-%dT%H:%M:%SZ")
    batch_id = f"{batch_ts.strftime('%Y%m%dT%H%M%SZ')}_{_batch_tag(args.scenario, args.group)}"
    if not args.dry_run:
        _ensure_complete_timing_detail(TIMING_XLSX, args.output_base)
    batch_t0 = time.perf_counter()
    for i, plan in enumerate(plans):
        if args.max_runs and i >= args.max_runs:
            break
        timing = execute_plan(
            plan,
            output_base=args.output_base,
            img_root=args.img_root,
            pose_dir=args.pose_dir,
            dry_run=args.dry_run,
            skip_existing=args.skip_existing,
            skip_llm=args.skip_llm,
            force_vision=args.force_vision,
            steps=step_set,
        )
        timing_rows.append(timing)
        counts[timing.status] = counts.get(timing.status, 0) + 1
        if not args.dry_run:
            _append_timing_run(
                TIMING_XLSX,
                _timing_row_dict(
                    timing,
                    batch_started_at=batch_started_at,
                    batch_id=batch_id,
                    steps=steps_label,
                ),
            )

    batch_wall_sec = time.perf_counter() - batch_t0

    print("\n=== Summary ===")
    for key, n in sorted(counts.items()):
        print(f"  {key}: {n}")

    if not args.dry_run and timing_rows:
        _print_batch_timing(timing_rows, batch_wall_sec)
        print(f"  timing report: {TIMING_XLSX}")


if __name__ == "__main__":
    main()
