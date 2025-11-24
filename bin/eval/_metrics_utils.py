from __future__ import annotations

import math
import re
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_VALIDITY_COMPONENTS: Sequence[str] = (
    "formula_validity",
    "spacegroup_validity",
    "site_multiplicity_validity",
    "bond_length_validity",
)


def _normalise_bool_series(series: pd.Series) -> pd.Series:
    """Return a numeric boolean series while preserving missing values."""
    if series.empty:
        return series.astype("boolean")

    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    bool_series = numeric.astype("boolean")
    return bool_series


def summarise_boolean_series(series: pd.Series) -> Tuple[int, int, float]:
    """Compute count/total/rate statistics for a boolean-like Series."""
    boolean = _normalise_bool_series(series).dropna()
    total = int(boolean.count())
    if total == 0:
        return 0, 0, float("nan")
    count = int(boolean.sum())
    rate = float(count / total) if total else float("nan")
    return count, total, rate


def compute_validity_breakdown(
    frame: pd.DataFrame,
    *,
    validity_column: str = "validity",
    components: Iterable[str] = DEFAULT_VALIDITY_COMPONENTS,
) -> Dict[str, object]:
    """Return aggregated validity metrics with per-component rates."""
    metrics: Dict[str, object] = {
        "validity_rate": float("nan"),
        "valid_count": 0,
        "valid_total": 0,
    }

    if validity_column in frame.columns:
        count, total, rate = summarise_boolean_series(frame[validity_column])
        metrics.update({"validity_rate": rate, "valid_count": count, "valid_total": total})

    component_columns = list(dict.fromkeys(components))
    component_columns.extend(
        column
        for column in frame.columns
        if column.endswith("_validity") and column not in component_columns
    )
    for column in component_columns:
        if column not in frame.columns:
            continue
        count, total, rate = summarise_boolean_series(frame[column])
        metrics[f"{column}_count"] = count
        metrics[f"{column}_total"] = total
        metrics[f"{column}_rate"] = rate

    return metrics


def _normalise_mode_label(value: object) -> str:
    """Convert structure match modes to consistent snake_case labels."""
    if value is None:
        return "none"
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return "none"
    elif isinstance(value, float) and math.isnan(value):
        return "none"
    else:
        cleaned = str(value).strip()
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", cleaned).strip("_").lower()
    if not slug or slug == "nan":
        return "none"
    return slug


def compute_structure_match_breakdown(
    frame: pd.DataFrame,
    *,
    column: str = "structure_match_mode",
) -> Dict[str, object]:
    """Aggregate counts per structure match mode."""
    if column not in frame.columns:
        return {}

    normalised = frame[column].apply(_normalise_mode_label)
    counts = normalised.value_counts(dropna=False)
    total = int(counts.sum())
    metrics: Dict[str, object] = {f"{column}_total": total}

    if total == 0:
        return metrics

    for mode, count in counts.items():
        metrics[f"{column}_{mode}_count"] = int(count)
        metrics[f"{column}_{mode}_rate"] = float(count / total)

    return metrics
