#!/usr/bin/env python3
"""
Publication-style figures from per-instance evaluation data.

Methodological explanation belongs in the LaTeX caption, not inside the figure.

Output: evaluation/results/plots/*.png

Usage (from repo root):
  python evaluation/scripts/plot_thesis_figures.py
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from eval_common import (  # noqa: E402
    DETECTION_REPORT_XLSX,
    LOCALIZATION_REPORT_XLSX,
    OBJECT_MATCHING_REPORT_XLSX,
    SCENE_COUNT_REPORT_XLSX,
    TRIANGULATION_LOCALIZATION_REPORT_XLSX,
)

PLOTS_DIR = _SCRIPT_DIR.parent / "results" / "plots"

# Typography (pt)
FS_LABEL = 10
FS_TICK = 9
FS_LEGEND = 9
FS_PANEL = 10

# Layout (inches)
FIG_CDF = (4.8, 3.2)
FIG_BOX = (5.2, 3.2)
CDF_LW = 1.4
CDF_GUIDE_LW = 0.7
POINT_SIZE = 4
POINT_ALPHA = 0.5
BOX_ALPHA = 0.38
BOX_ALPHA_NEUTRAL = 0.48

# Muted palette shared by CDFs and boxplots
PALETTE = {
    "blue": "#4878A8",
    "blue_light": "#A8C0D4",
    "terra": "#C4826E",
    "terra_light": "#E0C4B8",
    "green": "#6B9E7A",
    "green_dark": "#5A9A6F",
    "neutral": "#7A8490",
}

LIGHTING_ORDER = ("day", "afternoon", "night")
LIGHTING_LABEL = {"day": "Day", "afternoon": "Afternoon", "night": "Night"}

CLASS_ORDER = ("person", "vehicle", "emergency_vehicle")
CLASS_LABEL = {
    "person": "Person",
    "vehicle": "Vehicle",
    "emergency_vehicle": "Emergency vehicle",
}
CLASS_ALIASES = {"normal_vehicle": "vehicle"}

# legend label, x-tick label
SCENE_CONFIGS: List[Tuple[str, str, str, str]] = [
    ("mono", "vision-only", "Single-UAV, vision", "Single-UAV,\nvision"),
    (
        "mono",
        "vision and audio",
        "Single-UAV, vision and audio",
        "Single-UAV,\nvision and audio",
    ),
    ("multi", "vision-only", "Multi-UAV, vision", "Multi-UAV,\nvision"),
    (
        "multi",
        "vision and audio",
        "Multi-UAV, vision and audio",
        "Multi-UAV,\nvision and audio",
    ),
]

YLABEL_PER_RUN_RELATIVE_COUNT = "Relative mean count error"
SCENE_BOX_COLORS = [
    PALETTE["blue"],
    PALETTE["blue_light"],
    PALETTE["terra"],
    PALETTE["terra_light"],
]
LIGHTING_BOX_COLORS = [
    PALETTE["blue_light"],
    PALETTE["terra_light"],
    PALETTE["green"],
]
POINT_NEUTRAL = PALETTE["neutral"]

COLORS = {
    "single_uav": PALETTE["blue"],
    "multi_uav": PALETTE["green_dark"],
    "person": PALETTE["blue"],
    "vehicle": PALETTE["terra"],
    "emergency_vehicle": PALETTE["green"],
}

def _apply_style() -> None:
    common = {
        "font.size": FS_LABEL,
        "axes.labelsize": FS_LABEL,
        "axes.titlesize": FS_PANEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEGEND,
        "axes.labelweight": "normal",
        "axes.titleweight": "normal",
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
    }
    plt.rcParams.update(common)

    try:
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "text.latex.preamble": r"\usepackage{amsmath}",
            }
        )
        fig, ax = plt.subplots(figsize=(0.5, 0.5))
        ax.text(0.5, 0.5, "test")
        fig.savefig(io.BytesIO(), format="png")
        plt.close(fig)
        print("Font: LaTeX (text.usetex=True)")
    except Exception:
        plt.close("all")
        plt.rcParams.update(
            {
                "text.usetex": False,
                "font.family": "serif",
                "font.serif": [
                    "Computer Modern Roman",
                    "CMU Serif",
                    "DejaVu Serif",
                ],
                "mathtext.fontset": "cm",
            }
        )
        print("Font: Computer Modern serif fallback (LaTeX unavailable)")


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.png"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)
    return [path]


def _style_axes(ax: plt.Axes, *, ygrid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ygrid:
        ax.yaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.35, color="0.55")
        ax.set_axisbelow(True)


def _norm_class(value: object) -> str:
    return CLASS_ALIASES.get(str(value).strip(), str(value).strip())


def _ordered_lightings(values: Iterable[str]) -> List[str]:
    present = set(values)
    out = [x for x in LIGHTING_ORDER if x in present]
    out.extend(sorted(present - set(out)))
    return out


def _cdf(values: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.sort(np.asarray(values, dtype=float))
    if arr.size == 0:
        return np.array([0.0]), np.array([0.0])
    y = np.arange(1, arr.size + 1) / arr.size
    return arr, y


def _read_excel(path: Path, sheet: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing report: {path} (run evaluation scripts first)")
    return pd.read_excel(path, sheet_name=sheet)


def _legend_below(
    fig: plt.Figure,
    ax: plt.Axes,
    *,
    ncol: int = 3,
    anchor_y: float = -0.22,
    bottom: float = 0.26,
    fontsize: float | None = None,
) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    fig.subplots_adjust(bottom=bottom)
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, anchor_y),
        ncol=ncol,
        frameon=False,
        fontsize=fontsize if fontsize is not None else FS_LEGEND,
        handlelength=1.4,
        columnspacing=1.0,
    )


def _boxplot_with_points(
    ax: plt.Axes,
    groups: List[np.ndarray],
    tick_labels: List[str],
    *,
    colors: List[str] | None = None,
    point_color: str = POINT_NEUTRAL,
) -> None:
    bp = ax.boxplot(
        groups,
        tick_labels=tick_labels,
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "0.15", "linewidth": 1.0},
        whiskerprops={"linewidth": 0.7, "color": "0.45"},
        capprops={"linewidth": 0.7, "color": "0.45"},
        boxprops={"linewidth": 0.7, "edgecolor": "0.45"},
    )
    if colors:
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(BOX_ALPHA)
    else:
        for patch in bp["boxes"]:
            patch.set_facecolor("0.88")
            patch.set_alpha(BOX_ALPHA_NEUTRAL)

    rng = np.random.default_rng(42)
    for i, vals in enumerate(groups, start=1):
        if len(vals) == 0:
            continue
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter,
            vals,
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            color=point_color,
            edgecolors="none",
            zorder=3,
        )


def plot_cdf_detection_recall(out_dir: Path) -> List[Path]:
    df = _read_excel(DETECTION_REPORT_XLSX, "per_image")
    df = df[df["class"].notna()].copy()
    df["class"] = df["class"].map(_norm_class)
    df = df[df["gt_count"] > 0].copy()
    df["recall"] = df["tp"] / (df["tp"] + df["fn"])

    fig, ax = plt.subplots(figsize=FIG_CDF)
    for cls in CLASS_ORDER:
        vals = df.loc[df["class"] == cls, "recall"].to_numpy()
        if vals.size == 0:
            continue
        x, y = _cdf(vals)
        ax.plot(x, y, label=CLASS_LABEL[cls], color=COLORS[cls], linewidth=CDF_LW)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Cumulative fraction")
    _style_axes(ax)
    _legend_below(fig, ax, ncol=3)
    return _save(fig, out_dir, "01_cdf_detection_recall")


def _cdf_percentiles(values: np.ndarray) -> Tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(values, 50)), float(np.percentile(values, 90))


def _draw_cdf_percentile_guides(ax: plt.Axes, values: np.ndarray, color: str) -> None:
    """Vertical guides and markers at P50 and P90 (no on-plot value labels)."""
    p50, p90 = _cdf_percentiles(values)
    for pct_x, y_ref in ((p50, 0.5), (p90, 0.9)):
        ax.axvline(
            pct_x,
            color=color,
            linestyle=(0, (4, 3)),
            linewidth=CDF_GUIDE_LW,
            alpha=0.9,
            zorder=2,
        )
        ax.plot(pct_x, y_ref, "o", color=color, markersize=3.5, zorder=4)


def _legend_label_with_percentiles(name: str, values: np.ndarray) -> str:
    p50, p90 = _cdf_percentiles(values)
    if values.size == 0:
        return name
    return f"{name}  (P50={p50:.1f} m, P90={p90:.1f} m)"


def plot_cdf_position_error_mono_vs_tri(out_dir: Path) -> List[Path]:
    mono = _read_excel(LOCALIZATION_REPORT_XLSX, "per_pair")
    tri = _read_excel(TRIANGULATION_LOCALIZATION_REPORT_XLSX, "per_correct_match")
    m_vals = mono["err_pos_m"].dropna().to_numpy()
    t_vals = tri["err_pos_m"].dropna().to_numpy()

    fig, ax = plt.subplots(figsize=FIG_CDF)
    ax.axhline(0.5, color="0.78", linestyle=(0, (4, 3)), linewidth=CDF_GUIDE_LW, alpha=0.9, zorder=1)
    ax.axhline(0.9, color="0.78", linestyle=(0, (4, 3)), linewidth=CDF_GUIDE_LW, alpha=0.9, zorder=1)

    if m_vals.size:
        x, y = _cdf(m_vals)
        ax.plot(
            x,
            y,
            label=_legend_label_with_percentiles("Single-UAV", m_vals),
            color=COLORS["single_uav"],
            linewidth=CDF_LW,
        )
        _draw_cdf_percentile_guides(ax, m_vals, COLORS["single_uav"])
    if t_vals.size:
        x, y = _cdf(t_vals)
        ax.plot(
            x,
            y,
            label=_legend_label_with_percentiles("Multi-UAV triangulation", t_vals),
            color=COLORS["multi_uav"],
            linewidth=CDF_LW,
        )
        _draw_cdf_percentile_guides(ax, t_vals, COLORS["multi_uav"])

    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Position error (m)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_xlim(left=0)
    _style_axes(ax)
    _legend_below(fig, ax, ncol=1, anchor_y=-0.18, bottom=0.28, fontsize=FS_LEGEND - 1)
    return _save(fig, out_dir, "02_cdf_position_error_mono_vs_tri")


def _scene_runs() -> pd.DataFrame:
    return _read_excel(SCENE_COUNT_REPORT_XLSX, "per_run")


def _scene_run_mean_relative_count_error(df: pd.DataFrame) -> pd.DataFrame:
    """One relative error per run: mean (pred - gt) / gt across classes (undefined rows skipped)."""
    return (
        df.groupby(["config", "source", "run_id"], as_index=False)["relative_count_error"]
        .mean()
        .rename(columns={"relative_count_error": "mean_relative_count_error"})
    )


def plot_boxplot_scene_count_all_classes(out_dir: Path) -> List[Path]:
    """Per-run mean relative count error (all classes), 4 configurations."""
    df = _scene_run_mean_relative_count_error(_scene_runs())
    tick_labels = [c[3] for c in SCENE_CONFIGS]
    groups = [
        df.loc[
            (df["config"] == cfg) & (df["source"] == src), "mean_relative_count_error"
        ].to_numpy()
        for cfg, src, _, _ in SCENE_CONFIGS
    ]

    fig, ax = plt.subplots(figsize=FIG_BOX)
    _boxplot_with_points(
        ax,
        groups,
        tick_labels,
        colors=SCENE_BOX_COLORS,
        point_color=PALETTE["blue"],
    )
    ax.axhline(0, color="0.45", linewidth=0.7, linestyle="--", zorder=1)
    ax.set_ylabel(YLABEL_PER_RUN_RELATIVE_COUNT)
    _style_axes(ax)
    fig.subplots_adjust(left=0.16)
    return _save(fig, out_dir, "03_boxplot_scene_count_all")


def plot_boxplot_matching_recall(out_dir: Path) -> List[Path]:
    df = _read_excel(OBJECT_MATCHING_REPORT_XLSX, "per_pair")
    df = df[df["recall"].notna()].copy()
    lightings = _ordered_lightings(df["lighting"].unique())
    tick_labels = [LIGHTING_LABEL[lit] for lit in lightings]
    groups = [df.loc[df["lighting"] == lit, "recall"].to_numpy() for lit in lightings]

    fig, ax = plt.subplots(figsize=FIG_BOX)
    _boxplot_with_points(
        ax,
        groups,
        tick_labels,
        colors=LIGHTING_BOX_COLORS[: len(groups)],
        point_color=PALETTE["blue"],
    )
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Lighting")
    ax.set_ylabel("Recall")
    _style_axes(ax)
    return _save(fig, out_dir, "04_boxplot_matching_recall")


PLOTTERS = {
    "01": plot_cdf_detection_recall,
    "02": plot_cdf_position_error_mono_vs_tri,
    "03": plot_boxplot_scene_count_all_classes,
    "04": plot_boxplot_matching_recall,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Publication-style dissertation figures.")
    parser.add_argument("--output-dir", type=Path, default=PLOTS_DIR)
    parser.add_argument(
        "--only", nargs="*", metavar="ID", help="Figure ids: 01–04."
    )
    args = parser.parse_args()
    _apply_style()

    selected = args.only if args.only else sorted(PLOTTERS.keys())
    written: List[Path] = []
    for fig_id in selected:
        if fig_id not in PLOTTERS:
            parser.error(f"Unknown figure id: {fig_id}. Choose from: {', '.join(PLOTTERS)}")
        written.extend(PLOTTERS[fig_id](args.output_dir))
        print(f"[{fig_id}] {written[-1].name}")

    print(f"\nWrote {len(written)} files to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
