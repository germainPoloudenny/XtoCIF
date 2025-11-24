#!/usr/bin/env python3
"""Render the deCIFer overview figure with a neural-network style diagram."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


FIG_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = FIG_DIR / "deCIFer_overview.png"


def _pxrd_trace(x: np.ndarray) -> np.ndarray:
    """Synthesize a PXRD-like trace using a sum of Gaussian peaks."""
    peaks = [
        (0.08, 0.020, 1.00),
        (0.17, 0.025, 0.85),
        (0.32, 0.030, 1.15),
        (0.46, 0.022, 0.95),
        (0.58, 0.018, 0.70),
        (0.71, 0.027, 0.90),
        (0.86, 0.015, 0.60),
    ]
    y = np.zeros_like(x)
    for mu, sigma, amp in peaks:
        y += amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    y += 0.08 * np.sin(18 * np.pi * x) ** 2
    y += 0.04 * np.cos(7 * np.pi * x)
    y -= y.min()
    y /= y.max()
    return 0.12 + 0.82 * y


def _draw_pxrd_panel(ax: plt.Axes) -> None:
    """Draw the PXRD input panel."""
    panel = FancyBboxPatch(
        (0.65, 1.2),
        3.3,
        4.4,
        boxstyle="round,pad=0.4",
        fc="#fff3e6",
        ec="#e5b780",
        linewidth=2.0,
    )
    ax.add_patch(panel)
    ax.text(
        2.3,
        5.25,
        "PXRD pattern",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color="#62351d",
    )
    ax.text(
        2.3,
        4.85,
        "Powder diffraction intensity vs. 2θ",
        ha="center",
        va="center",
        fontsize=12,
        color="#62351d",
    )

    # Axes lines
    ax.plot([0.95, 3.95], [1.55, 1.55], color="#4c4c4c", lw=1.5)
    ax.plot([0.95, 0.95], [1.55, 4.85], color="#4c4c4c", lw=1.5)
    ax.text(0.75, 3.2, "Intensity", rotation=90, fontsize=11, color="#4c4c4c")
    ax.text(2.45, 1.25, r"2θ", fontsize=11, color="#4c4c4c")

    # PXRD trace
    x = np.linspace(0, 1, 500)
    y = _pxrd_trace(x)
    ax.plot(0.95 + 3.0 * x, 1.55 + 3.1 * y, color="#c15d00", lw=2.3, solid_capstyle="round")


def _draw_encoder_panel(ax: plt.Axes) -> None:
    """Draw the PXRD encoder block with CNN + transformer motif."""
    encoder = FancyBboxPatch(
        (5.6, 1.2),
        3.8,
        4.4,
        boxstyle="round,pad=0.35",
        fc="#eef1ff",
        ec="#9ea9ff",
        linewidth=2.2,
    )
    ax.add_patch(encoder)
    ax.text(
        7.5,
        5.2,
        "PXRD encoder",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color="#1e237e",
    )
    ax.text(
        7.5,
        4.9,
        "1D CNN + Transformer",
        ha="center",
        va="center",
        fontsize=12,
        color="#1e237e",
    )

    # Stylized neural network: three fully-connected layers.
    layer_x = [6.1, 7.2, 8.3]
    layer_sizes = [4, 5, 4]
    node_coords = []
    for x, size in zip(layer_x, layer_sizes):
        y_positions = np.linspace(2.3, 4.1, size)
        coords = [(x, y) for y in y_positions]
        node_coords.append(coords)
        for cx, cy in coords:
            circ = Circle(
                (cx, cy),
                0.13,
                facecolor="#eef3ff",
                edgecolor="#4f64c8",
                linewidth=1.2,
            )
            ax.add_patch(circ)
    for left_layer, right_layer in zip(node_coords[:-1], node_coords[1:]):
        for (x0, y0) in left_layer:
            for (x1, y1) in right_layer:
                ax.plot(
                    [x0 + 0.15, x1 - 0.15],
                    [y0, y1],
                    color="#4f64c8",
                    alpha=0.5,
                    lw=1.0,
                )
    ax.text(7.2, 2.0, "Neural feature extractor", fontsize=11, color="#1e237e", ha="center")


def _draw_decoder_panel(ax: plt.Axes) -> None:
    """Draw the autoregressive CIF generator block."""
    decoder = FancyBboxPatch(
        (10.1, 1.2),
        3.8,
        4.4,
        boxstyle="round,pad=0.35",
        fc="#fff0f6",
        ec="#f0a3c0",
        linewidth=2.2,
    )
    ax.add_patch(decoder)
    ax.text(
        12.0,
        5.2,
        "Autoregressive CIF generator",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color="#7a1130",
    )
    ax.text(
        12.0,
        4.9,
        "Transformer decoder",
        ha="center",
        va="center",
        fontsize=12,
        color="#7a1130",
    )

    token_labels = ["<bos>", "Symmetry", "Lattice", "Atomic sites", "CIF"]
    for idx, label in enumerate(token_labels):
        x = 10.6 + 0.55 * idx
        y = 4.4 - 0.65 * idx
        width = 1.6
        height = 0.55
        token = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.08",
            fc="#ffffff",
            ec="#dd709a",
            linewidth=1.2,
        )
        ax.add_patch(token)
        ax.text(
            x + width / 2,
            y + height / 2,
            label,
            ha="center",
            va="center",
            fontsize=11,
            color="#7a1130",
        )
        if idx < len(token_labels) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + width, y + height / 2),
                    (x + width + 0.4, y + height / 2 - 0.25),
                    arrowstyle="-|>",
                    mutation_scale=12,
                    color="#dd709a",
                    linewidth=1.2,
                )
            )


def _draw_optional_descriptor(ax: plt.Axes) -> None:
    """Draw the optional crystal descriptor conditioning block."""
    descriptor = FancyBboxPatch(
        (6.6, 0.4),
        3.1,
        0.95,
        boxstyle="round,pad=0.25",
        fc="#e8fff6",
        ec="#4fb091",
        linewidth=1.8,
    )
    ax.add_patch(descriptor)
    ax.text(
        8.15,
        0.86,
        "Crystal descriptors (optional)",
        ha="center",
        va="center",
        fontsize=12,
        color="#1f6651",
    )
    ax.text(
        8.15,
        0.55,
        "[composition, symmetry tokens]",
        ha="center",
        va="center",
        fontsize=10,
        color="#1f6651",
    )
    ax.add_patch(
        FancyArrowPatch(
            (8.15, 1.35),
            (11.0, 1.8),
            arrowstyle="-|>",
            mutation_scale=16,
            color="#1f6651",
            linewidth=1.5,
            connectionstyle="arc3,rad=-0.15",
        )
    )


def _connect_blocks(ax: plt.Axes) -> None:
    """Draw connecting arrows between the major blocks."""
    arrow_color = "#3d2d84"
    ax.add_patch(
        FancyArrowPatch(
            (4.1, 3.4),
            (5.5, 3.4),
            arrowstyle="-|>",
            mutation_scale=18,
            color=arrow_color,
            linewidth=2.2,
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            (9.4, 3.4),
            (10.0, 3.4),
            arrowstyle="-|>",
            mutation_scale=18,
            color=arrow_color,
            linewidth=2.2,
        )
    )
    ax.text(
        9.0,
        3.7,
        "PXRD embeddings",
        fontsize=12,
        ha="center",
        color=arrow_color,
    )


def render_overview() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 6.2), dpi=220)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")

    backdrop = FancyBboxPatch(
        (0.3, 0.4),
        13.2,
        6.1,
        boxstyle="round,pad=0.6",
        fc="#f9edf8",
        ec="#ddb9f0",
        linewidth=2.5,
    )
    ax.add_patch(backdrop)

    ax.text(
        7.0,
        6.2,
        "deCIFer: PXRD-conditioned neural architecture",
        fontsize=18,
        ha="center",
        va="center",
        color="#3b164a",
        fontweight="bold",
    )
    ax.text(
        7.0,
        5.85,
        "PXRD signals are encoded and prepended to the CIF token stream for autoregressive generation.",
        fontsize=12,
        ha="center",
        va="center",
        color="#3b164a",
    )

    _draw_pxrd_panel(ax)
    _draw_encoder_panel(ax)
    _draw_decoder_panel(ax)
    _draw_optional_descriptor(ax)
    _connect_blocks(ax)

    fig.savefig(OUTPUT_PATH, bbox_inches="tight", transparent=True)
    plt.close(fig)


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    render_overview()
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
