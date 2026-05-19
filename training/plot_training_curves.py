"""
Genera figura de convergencia (memoria) desde results/training_logs/*_history.csv

Uso:
    python -m training.plot_training_curves --stem ae_n100_rho0.0_multik
"""

from __future__ import annotations

import argparse
import csv
import os

import matplotlib.pyplot as plt

TRAINING_LOG_DIR = "results/training_logs"
FIG_DIR = "results/figures/training"


def load_history(stem: str) -> list[dict]:
    path = os.path.join(TRAINING_LOG_DIR, f"{stem}_history.csv")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_history(stem: str, show: bool = False) -> str:
    rows = load_history(stem)
    epochs = [int(r["epoch"]) for r in rows]
    total = [float(r["loss_total"]) for r in rows]
    recon = [float(r["loss_recon"]) for r in rows]
    clas = [float(r["loss_class"]) for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, total, marker="o", label=r"$\mathcal{L}_{\mathrm{total}}$")
    ax.plot(epochs, recon, marker="s", label=r"$\mathcal{L}_{\mathrm{recon}}$ (MSE)")
    ax.plot(epochs, clas, marker="^", label=r"$\mathcal{L}_{\mathrm{class}}$ (BCE)")
    ax.set_xlabel("Época")
    ax.set_ylabel("Pérdida (media por paso)")
    ax.set_title(f"Convergencia SupervisedAE — {stem}")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()

    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, f"{stem}_convergence.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    pdf = os.path.splitext(out)[0] + ".pdf"
    fig.savefig(pdf, bbox_inches="tight")
    print(f"Figura guardada: {out}")
    print(f"Figura guardada: {pdf}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stem", required=True, help="p.ej. ae_n100_rho0.0_multik")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    plot_history(args.stem, show=args.show)


if __name__ == "__main__":
    main()
