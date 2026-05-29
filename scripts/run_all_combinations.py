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
"""

from __future__ import annotations

import argparse
import csv
import itertools
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fusion.matching.cross_view_match import resolve_notebook_assets, view_id_to_frame_id
from fusion.run_layout import build_run_id, fusion_dir, perception_dir, run_dir_path, sls_path

DATASETS = _REPO_ROOT / "datasets"
IMG_ROOT = DATASETS / "images"
LABEL_DIR = DATASETS / "labels"
POSE_DIR = _REPO_ROOT / "perception" / "vision" / "data" / "telemetryData"
AUDIO_DIR = _REPO_ROOT / "perception" / "audio" / "data" / "generated_audios"
TRANSCRIPTIONS_DIR = _REPO_ROOT / "perception" / "audio" / "data" / "transcriptions"
VISION_CACHE = _REPO_ROOT / "output" / ".cache" / "vision"

INTEGRATION = _REPO_ROOT / "perception" / "vision" / "code" / "integration_pipeline.py"
TRANSCRIBE = _REPO_ROOT / "perception" / "audio" / "code" / "transcribe_run.py"
CROSS_VIEW = _REPO_ROOT / "fusion" / "matching" / "cross_view_match.py"
LLM_ORCH = _REPO_ROOT / "fusion" / "llm" / "code" / "llm_orchestrator.py"
SLS_BUILDER = _REPO_ROOT / "fusion" / "sls" / "sls_builder.py"


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
    label_dir: Path,
    pose_dir: Path,
) -> Optional[dict]:
    """PNG/labels in datasets/ (flat 00000.png); telemetry in perception/vision/data."""
    view_id = img_view_id(image_num)
    return resolve_notebook_assets(
        view_id,
        img_root=img_root,
        label_dir=label_dir,
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
    label_dir: Path,
    pose_dir: Path,
    dry_run: bool,
    force_vision: bool,
) -> bool:
    cache = vision_cache_path(image_num)
    if not force_vision and cache.is_file():
        if not dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cache, dest)
        return True

    assets = resolve_assets(
        image_num, img_root=img_root, label_dir=label_dir, pose_dir=pose_dir
    )
    if assets is None:
        print(f"  [skip vision] missing assets for img{image_num} under {img_root}")
        return False

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
    if code != 0:
        return False
    if not dry_run and dest.is_file():
        cache.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dest, cache)
    return True


def run_is_complete(run_dir: Path, multi: bool) -> bool:
    return sls_path(run_dir).is_file()


def execute_plan(
    plan: RunPlan,
    *,
    output_base: Path,
    img_root: Path,
    label_dir: Path,
    pose_dir: Path,
    dry_run: bool,
    skip_existing: bool,
    skip_llm: bool,
    force_vision: bool,
    steps: set,
) -> str:
    run_dir = run_dir_path(
        plan.scenario, plan.run_id, group=plan.group, output_base=output_base
    )
    if skip_existing and run_dir.is_dir() and run_is_complete(run_dir, plan.multi):
        return "skipped_complete"

    active = (
        {"vision", "transcribe", "cross", "llm", "sls"}
        if "all" in steps
        else steps
    )

    status = "ok"
    print(f"\n=== {plan.scenario} / {plan.group} / {plan.run_id} ===")

    if "vision" in active:
        pdir = perception_dir(run_dir)
        if plan.multi:
            for n in plan.image_nums:
                dest = pdir / f"visual_{img_view_id(n)}.json"
                if not ensure_vision_json(
                    plan,
                    n,
                    dest,
                    img_root=img_root,
                    label_dir=label_dir,
                    pose_dir=pose_dir,
                    dry_run=dry_run,
                    force_vision=force_vision,
                ):
                    return "vision_failed"
        else:
            n = plan.image_nums[0]
            dest = pdir / "visual.json"
            if not ensure_vision_json(
                plan,
                n,
                dest,
                img_root=img_root,
                label_dir=label_dir,
                pose_dir=pose_dir,
                dry_run=dry_run,
                force_vision=force_vision,
            ):
                return "vision_failed"

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
                return "transcribe_failed"
        else:
            print(f"  [skip transcribe] missing {audio_wav}")
            return "audio_missing"

    if plan.multi and "cross" in active:
        code = run_cmd(
            [
                sys.executable,
                str(CROSS_VIEW),
                "--run-dir",
                str(run_dir),
                "--no-raw-assets",
            ],
            dry_run=dry_run,
        )
        if code != 0:
            return "cross_failed"

    if "llm" in active and not skip_llm:
        code = run_cmd(
            [sys.executable, str(LLM_ORCH), "--run-dir", str(run_dir)],
            dry_run=dry_run,
        )
        if code != 0:
            return "llm_failed"

    if "sls" in active:
        code = run_cmd(
            [sys.executable, str(SLS_BUILDER), "--run-dir", str(run_dir)],
            dry_run=dry_run,
        )
        if code != 0:
            return "sls_failed"

    return status


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
        help="Frame PNGs (default: datasets/images)",
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=LABEL_DIR,
        help="YOLO labels (default: datasets/labels)",
    )
    parser.add_argument(
        "--pose-dir",
        type=Path,
        default=POSE_DIR,
        help="Telemetry txt (default: perception/vision/data/telemetryData)",
    )
    parser.add_argument("--dry-run", action="store_true", help="List runs only")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip run if fusion/sls.json already exists",
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
    counts: dict = {}
    for i, plan in enumerate(plans):
        if args.max_runs and i >= args.max_runs:
            break
        result = execute_plan(
            plan,
            output_base=args.output_base,
            img_root=args.img_root,
            label_dir=args.label_dir,
            pose_dir=args.pose_dir,
            dry_run=args.dry_run,
            skip_existing=args.skip_existing,
            skip_llm=args.skip_llm,
            force_vision=args.force_vision,
            steps=step_set,
        )
        counts[result] = counts.get(result, 0) + 1

    print("\n=== Summary ===")
    for key, n in sorted(counts.items()):
        print(f"  {key}: {n}")


if __name__ == "__main__":
    main()
