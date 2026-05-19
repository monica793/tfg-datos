"""
Figuras para la memoria (solo P_FA, P_MD, P_IE — sin P_global).
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np


METRIC_LABELS = ("P_FA", "P_MD", "P_IE")
METRIC_KEYS = ("p_fa", "p_md", "p_ie")


def plot_alpha_ablation_bars(
    results_by_alpha: dict[float, dict[str, float]],
    *,
    title: str,
    save_path: str,
    alpha_labels: dict[float, str] | None = None,
    show: bool = False,
    dpi: int = 200,
) -> str:
    """
    Barras agrupadas: dos configuraciones de alpha_mix, tres métricas.

    results_by_alpha: {alpha: {"p_fa", "p_md", "p_ie", "p_fa_lo", ...}}
    """
    alphas = sorted(results_by_alpha.keys())
    if len(alphas) != 2:
        raise ValueError("Esta figura de la memoria espera exactamente dos valores de alpha_mix.")

    if alpha_labels is None:
        alpha_labels = {
            0.0: r"$\alpha_{\mathrm{mix}}=0$ (Polar sobre $\mathbf{y}$)",
            1.0: r"$\alpha_{\mathrm{mix}}=1$ (Polar sobre $\hat{\mathbf{y}}$)",
        }

    x = np.arange(len(METRIC_LABELS))
    width = 0.36
    colors = ("#2c6eab", "#c44e52")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, alpha in enumerate(alphas):
        row = results_by_alpha[alpha]
        vals = [row[k] for k in METRIC_KEYS]
        yerr_low = [row[k] - row[f"{k}_lo"] for k in METRIC_KEYS]
        yerr_high = [row[f"{k}_hi"] - row[k] for k in METRIC_KEYS]
        yerr = np.array([yerr_low, yerr_high])
        offset = (i - 0.5) * width
        ax.bar(
            x + offset,
            vals,
            width,
            label=alpha_labels.get(alpha, f"alpha_mix={alpha:g}"),
            color=colors[i],
            yerr=yerr,
            capsize=4,
            error_kw={"elinewidth": 1.2},
        )

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_LABELS)
    ax.set_ylabel("Probabilidad")
    ax.set_title(title)
    ax.grid(True, which="both", axis="y", alpha=0.35)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    base, _ = os.path.splitext(save_path)
    for ext in (".png", ".pdf"):
        path = base + ext
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"Figura guardada: {path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return base + ".png"
