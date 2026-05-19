"""
Sección 4.2 — Validación de alpha_mix en el receptor híbrido.

Compara solo alpha_mix=0 y alpha_mix=1 en un punto de trabajo fijo (n, k, rho).
La detección de actividad usa el mismo p_active en ambos casos; solo cambia
la señal que alimenta demapper + Polar.

Uso (desde la raíz del repo):
    python -m evaluation.chapter4.s4_2_alpha_ablation
    python -m evaluation.chapter4.s4_2_alpha_ablation --rho-db 3 --k 50
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

# Raíz del proyecto en sys.path cuando se ejecuta como script
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from evaluation.chapter4.config import (
    BATCH_SIZE_SIM,
    CHAPTER4_RESULTS_DIR,
    N_BATCHES_EVAL,
    N_BLOCK,
    P_EMPTY,
    RHO_DBS_DEFAULT,
    TAU,
)
from evaluation.chapter4.plots import plot_alpha_ablation_bars
from evaluation.plot_pfa_pmd_pie import eval_pfa_pmd_pie, k_is_valid_for_5g
from systems.hybrid_polar import ActivityAwarePolarSystem
from training.train_hybrid import K_CAND, train_ae_for_n
from utils.signal import rho_db_to_ebno_db

SECTION_DIR = os.path.join(CHAPTER4_RESULTS_DIR, "s4_2_alpha_ablation")
ALPHAS_ABLATION = (0.0, 1.0)
DEFAULT_K = 50


def _valid_ks_for_n(n: int) -> list[int]:
    return [
        k for k in K_CAND
        if k < n and k_is_valid_for_5g(k, n) and not (12 <= k <= 19)
    ]


def load_multik_ae(n: int, rho_db: float):
    valid_ks = _valid_ks_for_n(n)
    if not valid_ks:
        raise RuntimeError(f"No hay k válidos para n={n}.")
    return train_ae_for_n(n=n, rho_db=rho_db, valid_ks=valid_ks, use_wandb=False)


def evaluate_alpha(
    *,
    n: int,
    k: int,
    rho_db: float,
    alpha_mix: float,
    ae,
) -> dict[str, float]:
    rate = k / n
    ebno_db = rho_db_to_ebno_db(rho_db, rate)
    system = ActivityAwarePolarSystem(
        k=k,
        n=n,
        ae_model=ae,
        p_empty=P_EMPTY,
        thresh=TAU,
        alpha_mix=alpha_mix,
    )
    _, p_fa, p_md, p_ie, _p_global, p_fa_ci, p_md_ci, p_ie_ci, _ = eval_pfa_pmd_pie(
        system, n, k, ebno_db, return_ci=True
    )
    return {
        "alpha_mix": float(alpha_mix),
        "n": int(n),
        "k": int(k),
        "R": float(rate),
        "rho_db": float(rho_db),
        "ebno_db": float(ebno_db),
        "p_fa": float(p_fa),
        "p_md": float(p_md),
        "p_ie": float(p_ie),
        "p_fa_lo": float(p_fa_ci[0]),
        "p_fa_hi": float(p_fa_ci[1]),
        "p_md_lo": float(p_md_ci[0]),
        "p_md_hi": float(p_md_ci[1]),
        "p_ie_lo": float(p_ie_ci[0]),
        "p_ie_hi": float(p_ie_ci[1]),
        "n_batches": int(N_BATCHES_EVAL),
        "batch_size": int(BATCH_SIZE_SIM),
    }


def run_ablation(
    *,
    n: int = N_BLOCK,
    k: int = DEFAULT_K,
    rho_db: float = 0.0,
    output_dir: str = SECTION_DIR,
    show: bool = False,
) -> dict[float, dict[str, float]]:
    if not k_is_valid_for_5g(k, n):
        raise ValueError(f"k={k} no es válido para Polar5G con n={n}.")

    os.makedirs(output_dir, exist_ok=True)
    ae = load_multik_ae(n, rho_db)

    results: dict[float, dict[str, float]] = {}
    csv_path = os.path.join(output_dir, f"metrics_n{n}_k{k}_rho{rho_db}.csv")
    fieldnames = [
        "alpha_mix", "n", "k", "R", "rho_db", "ebno_db",
        "p_fa", "p_md", "p_ie",
        "p_fa_lo", "p_fa_hi", "p_md_lo", "p_md_hi", "p_ie_lo", "p_ie_hi",
        "n_batches", "batch_size",
    ]

    print(
        f"\n=== Sección 4.2 | alpha ablation | n={n} | k={k} | rho={rho_db} dB "
        f"| {N_BATCHES_EVAL}x{BATCH_SIZE_SIM} ventanas ===\n"
    )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for alpha in ALPHAS_ABLATION:
            row = evaluate_alpha(n=n, k=k, rho_db=rho_db, alpha_mix=alpha, ae=ae)
            results[alpha] = row
            writer.writerow(row)
            print(
                f"alpha_mix={alpha:g} | P_FA={row['p_fa']:.3e} "
                f"[{row['p_fa_lo']:.3e}, {row['p_fa_hi']:.3e}] | "
                f"P_MD={row['p_md']:.3e} [{row['p_md_lo']:.3e}, {row['p_md_hi']:.3e}] | "
                f"P_IE={row['p_ie']:.3e} [{row['p_ie_lo']:.3e}, {row['p_ie_hi']:.3e}]"
            )

    print(f"\nTabla guardada: {csv_path}")

    fig_stem = os.path.join(output_dir, f"fig_alpha_ablation_n{n}_k{k}_rho{rho_db}")
    plot_alpha_ablation_bars(
        results,
        title=(
            f"Receptor híbrido: efecto de $\\alpha_{{\\mathrm{{mix}}}}$ "
            f"($n$={n}, $k$={k}, $\\rho$={rho_db:g} dB, $P_{{\\mathrm{{empty}}}}$={P_EMPTY})"
        ),
        save_path=fig_stem + ".png",
        show=show,
    )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Sección 4.2 — ablación alpha_mix (0 vs 1).")
    parser.add_argument("--n", type=int, default=N_BLOCK)
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="k del punto de trabajo (default: 50).")
    parser.add_argument(
        "--rho-db",
        type=float,
        default=None,
        help="SNR por símbolo en dB. Si se omite, se ejecuta para cada valor en RHO_DBS_DEFAULT.",
    )
    parser.add_argument("--output-dir", type=str, default=SECTION_DIR)
    parser.add_argument("--show", action="store_true", help="Mostrar figura interactiva.")
    args = parser.parse_args()

    rho_list = [args.rho_db] if args.rho_db is not None else list(RHO_DBS_DEFAULT)
    for rho_db in rho_list:
        run_ablation(
            n=args.n,
            k=args.k,
            rho_db=float(rho_db),
            output_dir=args.output_dir,
            show=args.show,
        )


if __name__ == "__main__":
    main()
