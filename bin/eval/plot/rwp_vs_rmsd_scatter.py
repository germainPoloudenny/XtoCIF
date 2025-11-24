#!/usr/bin/env python3
"""Plot Rwp ↔ RMSD scatter charts for beam-search candidates."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bin.eval.collect_evaluations import process as collect_evaluations


@dataclass(frozen=True)
class DatasetSpec:
    """Holds the label/path pair provided via --input."""

    label: str
    path: Path


def _parse_dataset_spec(arg: str) -> DatasetSpec:
    """Parse `label=path` assignments passed through --input."""

    if "=" not in arg:
        raise argparse.ArgumentTypeError(
            f"Expected --input entries to use the form label=path, got: {arg}"
        )

    label, path_str = arg.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("Dataset labels must be non-empty.")

    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Evaluation folder does not exist: {path}")
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"Expected a directory, got: {path}")
    return DatasetSpec(label=label, path=path)


def _select_sample_column(frame: pd.DataFrame) -> str:
    """Choose the column used to group hypotheses belonging to the same CIF."""

    if "index" in frame.columns:
        return "index"
    if "cif_name" in frame.columns:
        return "cif_name"
    raise ValueError(
        "Unable to determine the per-sample grouping column. "
        "Expected either 'index' or 'cif_name' in the collected DataFrame."
    )


def _load_dataset(
    spec: DatasetSpec,
    *,
    top_k: int,
    debug_max: int,
    max_samples: int | None,
) -> pd.DataFrame:
    """Load and preprocess evaluation rows for a dataset."""

    raw = collect_evaluations(
        spec.path,
        debug_max=None if debug_max <= 0 else debug_max,
        top_k=None if top_k <= 0 else top_k,
    )
    if raw.empty:
        raise RuntimeError(f"No evaluation rows found under {spec.path}")

    sample_col = _select_sample_column(raw)
    numeric = raw.copy()
    numeric["rwp"] = pd.to_numeric(numeric["rwp"], errors="coerce")
    numeric["rmsd"] = pd.to_numeric(numeric["rmsd"], errors="coerce")
    numeric["rwp"].replace([np.inf, -np.inf], np.nan, inplace=True)
    numeric["rmsd"].replace([np.inf, -np.inf], np.nan, inplace=True)
    numeric = numeric.dropna(subset=["rwp", "rmsd"])
    if numeric.empty:
        raise RuntimeError(f"All rows in {spec.path} have NaN/Inf Rwp or RMSD values.")

    if max_samples is not None and max_samples > 0:
        order = numeric[sample_col].drop_duplicates()
        keep = set(order.iloc[:max_samples])
        numeric = numeric[numeric[sample_col].isin(keep)]
        if numeric.empty:
            raise RuntimeError(
                f"After limiting to the first {max_samples} samples there were no rows left."
            )

    numeric = numeric[["rwp", "rmsd", sample_col]].copy()
    numeric.rename(columns={sample_col: "sample_id"}, inplace=True)
    numeric["dataset"] = spec.label
    return numeric.reset_index(drop=True)


def _compute_correlation(
    frame: pd.DataFrame, method: str
) -> tuple[float | None, int, int]:
    """Return (correlation, num_points, num_samples)."""

    if len(frame) < 2:
        return None, len(frame), frame["sample_id"].nunique()

    corr = frame["rwp"].corr(frame["rmsd"], method=method)
    return corr if pd.notna(corr) else None, len(frame), frame["sample_id"].nunique()


def _configure_axes_grid(num_axes: int, figsize: Tuple[float, float] | None) -> tuple[plt.Figure, List[plt.Axes]]:
    """Create a row of subplots with sensible defaults."""

    if num_axes <= 0:
        raise ValueError("At least one dataset must be provided.")

    width, height = figsize if figsize is not None else (4.8 * num_axes, 4.0)
    fig, axes = plt.subplots(
        1,
        num_axes,
        figsize=(width, height),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    return fig, axes[0].tolist()


def _annotate_axis(ax: plt.Axes, text: str) -> None:
    """Write an annotation in the top-left corner of the provided axis."""

    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox=dict(
            boxstyle="round,pad=0.2",
            facecolor="white",
            edgecolor="none",
            alpha=0.8,
        ),
    )


def plot_rwp_vs_rmsd(
    datasets: Iterable[pd.DataFrame],
    labels: Iterable[str],
    *,
    output_path: Path,
    corr_method: str,
    rmsd_threshold: float | None,
    rwp_threshold: float | None,
    alpha: float,
    marker_size: float,
    figsize: Tuple[float, float] | None,
    dpi: int,
    title: str | None,
) -> None:
    """Render the scatter plots and save them to disk."""

    data_frames = list(datasets)
    labels = list(labels)
    if len(data_frames) != len(labels):
        raise ValueError("datasets and labels must have the same length.")

    fig, axes = _configure_axes_grid(len(data_frames), figsize)
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4"])

    for idx, (frame, label, ax) in enumerate(zip(data_frames, labels, axes)):
        color = colors[idx % len(colors)]
        ax.scatter(
            frame["rwp"].to_numpy(),
            frame["rmsd"].to_numpy(),
            s=marker_size,
            alpha=alpha,
            edgecolors="none",
            color=color,
        )

        if rmsd_threshold is not None:
            ax.axhline(
                rmsd_threshold,
                color="tab:gray",
                linestyle="--",
                linewidth=1.0,
                alpha=0.8,
            )
        if rwp_threshold is not None:
            ax.axvline(
                rwp_threshold,
                color="tab:gray",
                linestyle="--",
                linewidth=1.0,
                alpha=0.8,
            )

        ax.set_title(label)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
        corr, num_points, num_samples = _compute_correlation(frame, corr_method)
        corr_str = "n/a" if corr is None else f"{corr:.3f}"
        annotation = (
            f"{corr_method.title()} r = {corr_str}\n"
            f"{num_points} candidates / {num_samples} CIFs"
        )
        _annotate_axis(ax, annotation)

    for ax in axes:
        ax.set_xlim(left=0.0)
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel(r"$R_{\mathrm{wp}}$")
    axes[0].set_ylabel("RMSD (Å)")

    fig.tight_layout()
    if title:
        fig.suptitle(title, y=1.02, fontsize=14)
        fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Scatter plot written to {output_path}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render Rwp ↔ RMSD scatter plots for the top-K beam-search candidates "
            "stored in evaluation folders."
        )
    )
    parser.add_argument(
        "--input",
        action="append",
        type=_parse_dataset_spec,
        required=True,
        metavar="LABEL=PATH",
        help=(
            "Dataset specification. Provide one entry per dataset, e.g. "
            "--input NOMA=/path/to/eval. The path must contain .pkl.gz files "
            "written by evaluate.py."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path where the resulting figure will be written (PNG/PDF/SVG).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help=(
            "Keep the top-K hypotheses per CIF ranked by Rwp before plotting. "
            "Set to 0 to keep every hypothesis present in the evaluation folder."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on the number of CIFs per dataset (applied after top-k).",
    )
    parser.add_argument(
        "--debug-max",
        type=int,
        default=0,
        help="Limit the number of .pkl.gz files read per dataset for quick iterations.",
    )
    parser.add_argument(
        "--corr-method",
        choices=["pearson", "spearman", "kendall"],
        default="spearman",
        help="Correlation statistic reported inside each subplot (default: spearman).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.35,
        help="Point transparency for the scatter plot.",
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        default=10.0,
        help="Marker size passed to matplotlib.pyplot.scatter.",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="Figure size in inches. Defaults to 4.8\" per panel by 4\" tall.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Resolution used when saving the figure (default: 200).",
    )
    parser.add_argument(
        "--rmsd-threshold",
        type=float,
        default=0.5,
        help="Optional horizontal reference line marking the RMSD match threshold.",
    )
    parser.add_argument(
        "--rwp-threshold",
        type=float,
        default=None,
        help="Optional vertical reference line showing an Rwp acceptance threshold.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional overall title placed above the grid of plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    frames: List[pd.DataFrame] = []
    labels: List[str] = []

    for spec in args.input:
        frame = _load_dataset(
            spec,
            top_k=args.top_k,
            debug_max=args.debug_max,
            max_samples=args.max_samples,
        )
        frames.append(frame)
        labels.append(spec.label)
        unique_samples = frame["sample_id"].nunique()
        print(
            f"Loaded {len(frame):,} candidates spanning {unique_samples:,} CIFs "
            f"from {spec.path} ({spec.label})."
        )

    plot_rwp_vs_rmsd(
        frames,
        labels,
        output_path=args.output,
        corr_method=args.corr_method,
        rmsd_threshold=args.rmsd_threshold,
        rwp_threshold=args.rwp_threshold,
        alpha=args.alpha,
        marker_size=args.marker_size,
        figsize=tuple(args.figsize) if args.figsize is not None else None,
        dpi=args.dpi,
        title=args.title,
    )


if __name__ == "__main__":
    main()
