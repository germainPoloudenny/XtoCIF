#!/usr/bin/env python3
"""Plot a Pareto chart of RMSD failure causes from evaluation outputs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import textwrap

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bin.eval.collect_evaluations import process as collect_evaluations


@dataclass(frozen=True)
class DatasetSpec:
    label: str
    path: Path


def _parse_dataset_spec(arg: str) -> DatasetSpec:
    if "=" in arg:
        label, path_str = arg.split("=", 1)
        label = label.strip() or "dataset"
        path = Path(path_str).expanduser().resolve()
    else:
        path = Path(arg).expanduser().resolve()
        label = path.name
    if not path.exists() or not path.is_dir():
        raise argparse.ArgumentTypeError(f"Evaluation folder not found: {path}")
    return DatasetSpec(label=label, path=path)


CAUSE_LABELS: Dict[str, str] = {
    "atom_count_mismatch": "Atom count mismatch",
    "composition_mismatch": "Composition mismatch",
    "geometry_mismatch": "Geometric incompatibility",
    "invalid_structure": "Invalid structure",
    "unknown": "Unknown",
}


def _read_aggregated_pickles_from_dir(path: Path) -> pd.DataFrame | None:
    """Try to read one or more aggregated DataFrames from a directory.

    Returns a concatenated DataFrame or ``None`` if none were readable.
    """
    frames: List[pd.DataFrame] = []
    for p in sorted(path.glob("*.pkl")) + sorted(path.glob("*.pkl.gz")):
        try:
            obj = pd.read_pickle(p)
        except Exception:
            continue
        if isinstance(obj, pd.DataFrame):
            frames.append(obj)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _apply_topk(frame: pd.DataFrame, top_k: int) -> pd.DataFrame:
    if top_k <= 0:
        return frame
    metric_column = "rwp" if "rwp" in frame.columns else None
    if metric_column is None:
        return frame
    if "index" in frame.columns:
        group_col = "index"
    elif "cif_name" in frame.columns:
        group_col = "cif_name"
    else:
        return frame
    sort_cols = [group_col, metric_column]
    if "rep" in frame.columns:
        sort_cols.append("rep")
    return (
        frame.sort_values(by=sort_cols, ascending=True)
        .groupby(group_col, group_keys=False)
        .head(top_k)
        .reset_index(drop=True)
    )


def _load_topk(spec: DatasetSpec, *, top_k: int, debug_max: int) -> pd.DataFrame:
    """Load evaluation rows from a folder or aggregated pickle.

    Supports two input forms:
    - Directory of per-sample pickles → uses collect_evaluations.process
    - Aggregated pickle(s) (.pkl/.pkl.gz) → reads with pandas and applies top-k
    """
    if spec.path.is_file():
        df = pd.read_pickle(spec.path)
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise RuntimeError(f"Unsupported pickle format: {spec.path}")
        df = _apply_topk(df, top_k)
    else:
        try:
            df = collect_evaluations(
                spec.path,
                debug_max=None if debug_max <= 0 else debug_max,
                top_k=None if top_k <= 0 else top_k,
            )
        except Exception:
            # Directory may contain pre-aggregated DataFrames
            df = _read_aggregated_pickles_from_dir(spec.path)
            if df is None or df.empty:
                raise
            df = _apply_topk(df, top_k)

    if df.empty:
        raise RuntimeError(f"No evaluation rows found under {spec.path}")
    df["rmsd"] = pd.to_numeric(df.get("rmsd"), errors="coerce")
    return df


def _failure_cause_counts(df: pd.DataFrame) -> Tuple[List[str], np.ndarray, int]:
    """Return (raw_cause_labels, counts, total_failures)."""
    fail_mask = ~np.isfinite(df["rmsd"])  # NaN/Inf RMSD → non-match
    failures = df.loc[fail_mask]
    total = int(failures.shape[0])
    if total == 0:
        return [], np.array([], dtype=int), 0
    series = failures.get("rmsd_failure_cause", pd.Series(["unknown"] * total))
    series = series.fillna("unknown").astype(str)
    counts = series.value_counts(dropna=False)
    labels = counts.index.tolist()
    values = counts.to_numpy()
    return labels, values, total


def _collapse_small(labels: List[str], counts: np.ndarray, *, min_share: float, min_count: int) -> Tuple[List[str], np.ndarray]:
    total = counts.sum()
    keep_mask = (counts >= max(1, min_count)) & ((counts / total) >= min_share)
    if keep_mask.all():
        return labels, counts
    keep_labels = [lab for lab, keep in zip(labels, keep_mask) if keep]
    keep_counts = counts[keep_mask]
    other_count = int(counts[~keep_mask].sum())
    if other_count > 0:
        keep_labels.append("Other")
        keep_counts = np.concatenate([keep_counts, np.array([other_count])])
    return keep_labels, keep_counts


def _sort_counts(labels: List[str], counts: np.ndarray, *, ascending: bool) -> Tuple[List[str], np.ndarray]:
    order = np.argsort(counts)
    if not ascending:
        order = order[::-1]
    return [labels[i] for i in order], counts[order]


def _prettify_labels(labels: List[str]) -> List[str]:
    out = []
    for lab in labels:
        out.append(CAUSE_LABELS.get(lab, lab))
    return out


def _wrap_labels(labels: List[str], *, width: int) -> List[str]:
    """Optionally line-wrap category labels for compact x-axis.

    Wrapping avoids mid-word breaks for readability.
    width <= 0 means no wrapping.
    """
    if width and width > 0:
        return [
            textwrap.fill(
                lab,
                width=width,
                break_long_words=False,
                break_on_hyphens=True,
            )
            for lab in labels
        ]
    return labels


def _plot_pareto(
    labels: List[str],
    counts: np.ndarray,
    *,
    title: str | None,
    dpi: int,
    figsize: Tuple[float, float],
    output: Path,
    xrot: int = 0,
    annotate: str = "both",  # one of: none|count|pct|both
    bar_color: str = "#9ecae1",
    line_color: str = "#25507a",
    wrap: int = 0,
    font_size: int = 9,
    tick_size: int = 8,
    line_point_labels: str = "end",  # 'all' | 'end' | 'none'
    inline_name: bool = False,
    line_label_offset: float = 2.0,   # data-unit fallback
    line_label_offset_pt: float = 4.0,  # preferred offset in points
    cum_headroom: float = 0.0,
    show_80_label: bool = False,
) -> None:
    total = int(counts.sum())
    x = np.arange(len(labels))
    cum_pct = counts.cumsum() / total * 100.0 if total > 0 else np.zeros_like(counts, dtype=float)

    # Slightly larger default font for print-style figures
    plt.rcParams.update({
        "font.size": font_size,
        "axes.titlesize": max(font_size, 9),
        "axes.labelsize": font_size,
        "xtick.labelsize": tick_size,
        "ytick.labelsize": tick_size,
    })

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    labels = _wrap_labels(labels, width=wrap)

    bars = ax.bar(
        x,
        counts,
        color=bar_color,
        alpha=0.6,
        edgecolor=line_color,
        linewidth=1.0,
    )

    y_off = max(1, int(np.max(counts) * 0.02)) if len(counts) else 1
    for i, v in enumerate(counts):
        if annotate == "none":
            continue
        pct = (v / total * 100.0) if total else 0.0
        if annotate == "count":
            txt = f"{int(v)}"
        elif annotate == "pct":
            txt = f"{pct:.0f}%"
        else:
            txt = f"{int(v)} ({pct:.0f}%)"
        ax.text(i, v + y_off, txt, ha="center", va="bottom", fontsize=8)

    ax.set_ylabel(f"Count (N={total})")
    ax.set_ylim(0, max(total, int(np.max(counts) * 1.15) if counts.size else 1))
    ax.set_xticks(x, labels, rotation=xrot, ha="right" if xrot else "center")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)

    ax2 = ax.twinx()
    ax2.plot(x, cum_pct, color=line_color, marker="o", markersize=5, linewidth=2.0)
    ax2.set_ylabel("Cumulative (%)")
    ax2.set_ylim(0, 100 + max(0, float(cum_headroom)))
    ax2.set_yticks([0, 20, 40, 60, 80, 100])
    ax2.axhline(80, color="gray", linestyle="--", linewidth=0.9)
    if show_80_label:
        ax2.text(len(x) - 0.5 if len(x) else 0, 80 + 2, "80%", color="gray", fontsize=8)

    if line_point_labels != "none":
        for i, p in enumerate(cum_pct):
            if line_point_labels == "end" and i != len(x) - 1:
                continue
            if line_label_offset_pt and line_label_offset_pt > 0:
                ax2.annotate(
                    f"{p:.0f}%",
                    (x[i], p),
                    xytext=(0, line_label_offset_pt),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    color=line_color,
                    fontsize=8,
                    clip_on=False,
                )
            else:
                top_cap = 100 + max(0, float(cum_headroom)) - 1.0
                y = min(top_cap, p + line_label_offset)
                ax2.text(i, y, f"{p:.0f}%", ha="center", va="bottom", color=line_color, fontsize=8)
    if inline_name and len(x):
        ax2.text(max(0, x[-1] - 0.25), min(100, cum_pct[-1] + 6), "Cumulative %", color=line_color, fontsize=9)

    if title:
        ax.set_title(title)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Pareto figure written to {output}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render a Pareto chart of non-match causes from evaluation results.")
    p.add_argument("--input", required=True, type=_parse_dataset_spec, metavar="[LABEL=]PATH", help="Evaluation folder containing .pkl.gz results.")
    p.add_argument("--output", required=True, type=Path, help="Where to write the figure (PNG/PDF/SVG).")
    p.add_argument("--title", default="", type=str, help="Optional figure title (default: none).")
    p.add_argument("--top-k", type=int, default=1, help="Keep the top-K hypotheses per CIF before counting failures (default: 1).")
    p.add_argument("--debug-max", type=int, default=0, help="Limit the number of .pkl.gz files read for quick iterations.")
    p.add_argument("--min-share", type=float, default=0.02, help="Group categories below this proportion into 'Other' (default: 0.02 = 2%).")
    p.add_argument("--min-count", type=int, default=1, help="Group categories with count < min-count into 'Other' (default: 1).")
    p.add_argument("--dpi", type=int, default=300, help="Output resolution (default: 300).")
    p.add_argument("--figsize", type=float, nargs=2, metavar=("WIDTH", "HEIGHT"), default=(7.2, 2.9), help="Figure size in inches (default: 7.2 2.9).")
    p.add_argument("--ascending", action="store_true", help="Sort bars in ascending order (default: descending).")
    p.add_argument("--xrot", type=int, default=0, help="Rotate x labels in degrees (default: 0).")
    p.add_argument("--annot", type=str, choices=["none", "count", "pct", "both"], default="both", help="Bar annotation mode (default: both).")
    p.add_argument("--wrap", type=int, default=10, help="Wrap x labels at N characters (0 = no wrap, default: 10).")
    p.add_argument("--bar-color", type=str, default="#9ecae1", help="Bar fill color (default: #9ecae1).")
    p.add_argument("--line-color", type=str, default="#25507a", help="Cumulative line color (default: #25507a).")
    p.add_argument("--font-size", type=int, default=9, help="Base font size for labels (default: 9).")
    p.add_argument("--tick-size", type=int, default=8, help="Tick label size (default: 8).")
    p.add_argument("--line-labels", type=str, choices=["all", "end", "none"], default="end", help="Where to place cumulative % labels (default: end).")
    p.add_argument("--inline-name", action="store_true", help="Show inline series label 'Cumulative %' (default: hidden).")
    p.add_argument("--line-label-offset", type=float, default=2.0, help="Vertical offset in data units for cumulative % labels (fallback mode).")
    p.add_argument("--line-label-offset-pt", type=float, default=4.0, help="Vertical offset in points for cumulative % labels (preferred).")
    p.add_argument("--cum-headroom", type=float, default=0.0, help="Extra headroom above 100% on right axis (default: 0).")
    p.add_argument("--show-80-label", action="store_true", help="Also write '80%' near the threshold line (default: hidden).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    spec: DatasetSpec = args.input

    df = _load_topk(spec, top_k=args.top_k, debug_max=args.debug_max)
    raw_labels, raw_counts, total_failures = _failure_cause_counts(df)

    if total_failures == 0:
        raise SystemExit("No non-match entries found (all RMSD values are finite). Nothing to plot.")

    # Collapse and sort
    labels, counts = _collapse_small(raw_labels, raw_counts, min_share=args.min_share, min_count=args.min_count)
    labels, counts = _sort_counts(labels, counts, ascending=args.ascending)
    labels = _prettify_labels(labels)

    # No title by default; use provided one if non-empty
    title = args.title
    _plot_pareto(
        labels,
        counts,
        title=title,
        dpi=args.dpi,
        figsize=tuple(args.figsize),
        output=args.output,
        xrot=args.xrot,
        annotate=args.annot,
        bar_color=args.bar_color,
        line_color=args.line_color,
        wrap=args.wrap,
        font_size=args.font_size,
        tick_size=args.tick_size,
        line_point_labels=args.line_labels,
        inline_name=args.inline_name,
        line_label_offset=args.line_label_offset,
        line_label_offset_pt=args.line_label_offset_pt,
        cum_headroom=args.cum_headroom,
        show_80_label=args.show_80_label,
    )


if __name__ == "__main__":
    main()
