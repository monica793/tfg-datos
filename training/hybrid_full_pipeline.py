"""
Orquestador integral de experimentos híbridos en una sola ejecución.

Genera:
- Entrenamiento/carga de checkpoints por configuración y rho.
- Barrido de tasa R=k/n.
- Barrido de alpha_mix.
- Barrido vs Eb/No.
- Barrido de umbral tau.

Todo se guarda bajo: results/runs/<run_id>/
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt

from evaluation.plot_pfa_pmd_pie import (
    eval_pfa_pmd_pie,
    k_is_valid_for_5g,
    run_curves_for_n,
    run_threshold_sweep,
)
from evaluation.plot_pie_vs_snr import eval_metrics_at_snr
from systems.hybrid_polar import ActivityAwarePolarSystem
from training.run_hybrid_experiments import EXPERIMENTS
from training.train_hybrid import K_CAND, N_FIXED, RHO_DBS, train_ae_for_n
from utils.signal import rho_db_to_ebno_db


def _default_valid_ks(n: int) -> list[int]:
    return [
        k for k in K_CAND
        if k < n and k_is_valid_for_5g(k, n) and not (12 <= k <= 19)
    ]


def _append_csv(path: str, fieldnames: list[str], row: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new_file = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            w.writeheader()
        w.writerow(row)


def _plot_alpha_sweep(alphas: list[float], pies: list[float], save_path: str, title: str, show: bool) -> None:
    plt.figure(figsize=(6.5, 5))
    plt.semilogy(alphas, pies, marker="o")
    plt.xlabel("alpha_mix (0=sin AE, 1=AE puro)")
    plt.ylabel("P_IE")
    plt.title(title)
    plt.grid(True, which="both", alpha=0.35)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Figura guardada: {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


def _plot_snr_three_panels(snrs: list[float], pfas: list[float], pmds: list[float], pies: list[float],
                           save_path: str, title: str, show: bool) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, vals, name in zip(axes, [pfas, pmds, pies], ["P_FA", "P_MD", "P_IE"]):
        ax.semilogy(snrs, vals, marker="o")
        ax.set_title(name)
        ax.set_xlabel("Eb/No (dB)")
        ax.grid(True, which="both", alpha=0.35)
    fig.suptitle(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Figura guardada: {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


def run_full_pipeline(
    run_id: str | None = None,
    n: int = N_FIXED,
    experiments: list[dict[str, Any]] | None = None,
    rho_dbs: list[float] | None = None,
    valid_ks: list[int] | None = None,
    ae_epochs: int = 20,
    ae_steps: int = 300,
    use_wandb: bool = False,
    alphas: list[float] | None = None,
    k_alpha: int = 50,
    k_snr: int = 50,
    ebno_range: list[float] | None = None,
    k_tau: int = 50,
    show_figures: bool = False,
    do_rate_sweep: bool = True,
    do_alpha_sweep: bool = True,
    do_snr_sweep: bool = True,
    do_tau_sweep: bool = True,
) -> dict[str, Any]:
    if experiments is None:
        experiments = EXPERIMENTS
    if rho_dbs is None:
        rho_dbs = list(RHO_DBS)
    if valid_ks is None:
        valid_ks = _default_valid_ks(n)
    if alphas is None:
        alphas = [0.0, 0.3, 0.5, 0.7, 1.0]
    if ebno_range is None:
        ebno_range = list(range(-2, 12))

    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join("results", "runs", run_id)
    fig_dir = os.path.join(base, "figures")
    tbl_dir = os.path.join(base, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tbl_dir, exist_ok=True)

    config = {
        "run_id": run_id,
        "n": n,
        "rho_dbs": rho_dbs,
        "valid_ks": valid_ks,
        "experiments": experiments,
        "alphas": alphas,
        "k_alpha": k_alpha,
        "k_snr": k_snr,
        "k_tau": k_tau,
        "ebno_range": ebno_range,
        "ae_epochs": ae_epochs,
        "ae_steps": ae_steps,
        "use_wandb": use_wandb,
        "show_figures": show_figures,
        "flags": {
            "do_rate_sweep": do_rate_sweep,
            "do_alpha_sweep": do_alpha_sweep,
            "do_snr_sweep": do_snr_sweep,
            "do_tau_sweep": do_tau_sweep,
        },
    }
    with open(os.path.join(base, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    summary: dict[str, Any] = {"run_id": run_id, "results": []}

    rate_fields = ["run_id", "exp_name", "rho_db", "k", "n", "R", "ebno_db", "p_fa", "p_md", "p_ie"]
    alpha_fields = ["run_id", "exp_name", "rho_db", "alpha_mix", "k", "n", "R", "ebno_db", "p_fa", "p_md", "p_ie"]
    snr_fields = ["run_id", "exp_name", "rho_db", "k", "n", "R", "ebno_db", "p_fa", "p_md", "p_ie"]
    tau_fields = ["run_id", "exp_name", "rho_db", "tau", "k", "n", "R", "ebno_db", "p_fa", "p_md", "p_ie"]

    for exp in experiments:
        exp_name = exp["name"]
        print(f"\n{'='*72}\n[PIPELINE] {exp_name}\n{'='*72}")

        trained: dict[float, Any] = {}
        for rho_db in rho_dbs:
            trained[float(rho_db)] = train_ae_for_n(
                n=n,
                rho_db=float(rho_db),
                valid_ks=valid_ks,
                use_wandb=use_wandb,
                checkpoint_tag=exp_name,
                latent_dim=exp.get("latent_dim", 64),
                hidden_dim=exp.get("hidden_dim", 256),
                dropout=exp.get("dropout", 0.1),
                ae_epochs=ae_epochs,
                ae_steps_per_epoch=ae_steps,
                w_recon=exp.get("w_recon", 1.0),
                w_class=exp.get("w_class", 10.0),
            )

        for rho_db, ae in trained.items():
            exp_result: dict[str, Any] = {"exp_name": exp_name, "rho_db": float(rho_db)}
            label = f"Hibrido_{exp_name}_rho{rho_db}"

            if do_rate_sweep:
                def make_sys_rate(k_: int, n_: int) -> ActivityAwarePolarSystem:
                    return ActivityAwarePolarSystem(k=k_, n=n_, ae_model=ae, p_empty=0.3, thresh=0.5)

                rate_fig = os.path.join(fig_dir, f"rate_{exp_name}_rho{rho_db}_n{n}.png")
                out = run_curves_for_n(
                    n=n,
                    rho_db=float(rho_db),
                    make_system=make_sys_rate,
                    label=label,
                    k_cand=valid_ks,
                    figure_path=rate_fig,
                    show_figure=show_figures,
                )
                if out:
                    rs, pfas, pmds, pies, _ = out
                    ks_eval = [k for k in valid_ks if k < n and k_is_valid_for_5g(k, n)]
                    best_idx = int(min(range(len(pies)), key=lambda i: pies[i]))
                    exp_result["rate_best_p_ie"] = float(pies[best_idx])
                    exp_result["rate_best_r"] = float(rs[best_idx])
                    exp_result["rate_best_k"] = int(ks_eval[best_idx]) if best_idx < len(ks_eval) else None
                    for i, k_ in enumerate(ks_eval):
                        if i >= len(rs):
                            break
                        ebno_db = rho_db_to_ebno_db(float(rho_db), k_ / n)
                        _append_csv(
                            os.path.join(tbl_dir, "rate_sweep.csv"),
                            rate_fields,
                            {
                                "run_id": run_id,
                                "exp_name": exp_name,
                                "rho_db": float(rho_db),
                                "k": int(k_),
                                "n": int(n),
                                "R": float(rs[i]),
                                "ebno_db": float(ebno_db),
                                "p_fa": float(pfas[i]),
                                "p_md": float(pmds[i]),
                                "p_ie": float(pies[i]),
                            },
                        )

            if do_alpha_sweep and (k_alpha < n) and k_is_valid_for_5g(k_alpha, n):
                r_alpha = k_alpha / n
                ebno_alpha = rho_db_to_ebno_db(float(rho_db), r_alpha)
                pies_alpha: list[float] = []
                for alpha in alphas:
                    sys_alpha = ActivityAwarePolarSystem(
                        k=k_alpha, n=n, ae_model=ae, p_empty=0.3, thresh=0.5, alpha_mix=float(alpha)
                    )
                    _, p_fa, p_md, p_ie, _ = eval_pfa_pmd_pie(
                        sys_alpha, n, k_alpha, ebno_alpha, return_ci=False
                    )
                    pies_alpha.append(float(p_ie))
                    _append_csv(
                        os.path.join(tbl_dir, "alpha_sweep.csv"),
                        alpha_fields,
                        {
                            "run_id": run_id,
                            "exp_name": exp_name,
                            "rho_db": float(rho_db),
                            "alpha_mix": float(alpha),
                            "k": int(k_alpha),
                            "n": int(n),
                            "R": float(r_alpha),
                            "ebno_db": float(ebno_alpha),
                            "p_fa": float(p_fa),
                            "p_md": float(p_md),
                            "p_ie": float(p_ie),
                        },
                    )

                best_alpha_idx = int(min(range(len(pies_alpha)), key=lambda i: pies_alpha[i]))
                exp_result["alpha_best"] = float(alphas[best_alpha_idx])
                exp_result["alpha_best_p_ie"] = float(pies_alpha[best_alpha_idx])
                _plot_alpha_sweep(
                    [float(a) for a in alphas],
                    pies_alpha,
                    save_path=os.path.join(fig_dir, f"alpha_{exp_name}_rho{rho_db}_k{k_alpha}.png"),
                    title=f"{label} | P_IE vs alpha_mix | k={k_alpha}, n={n}",
                    show=show_figures,
                )

            if do_snr_sweep and (k_snr < n) and k_is_valid_for_5g(k_snr, n):
                r_snr = k_snr / n
                snrs: list[float] = []
                pfas_snr: list[float] = []
                pmds_snr: list[float] = []
                pies_snr: list[float] = []
                sys_snr = ActivityAwarePolarSystem(k=k_snr, n=n, ae_model=ae, p_empty=0.3, thresh=0.5)
                for ebno_db in ebno_range:
                    p_fa, p_md, p_ie, _ = eval_metrics_at_snr(sys_snr, float(ebno_db))
                    snrs.append(float(ebno_db))
                    pfas_snr.append(float(p_fa))
                    pmds_snr.append(float(p_md))
                    pies_snr.append(float(p_ie))
                    _append_csv(
                        os.path.join(tbl_dir, "snr_sweep.csv"),
                        snr_fields,
                        {
                            "run_id": run_id,
                            "exp_name": exp_name,
                            "rho_db": float(rho_db),
                            "k": int(k_snr),
                            "n": int(n),
                            "R": float(r_snr),
                            "ebno_db": float(ebno_db),
                            "p_fa": float(p_fa),
                            "p_md": float(p_md),
                            "p_ie": float(p_ie),
                        },
                    )
                best_snr_idx = int(min(range(len(pies_snr)), key=lambda i: pies_snr[i]))
                exp_result["snr_best_ebno_db"] = float(snrs[best_snr_idx])
                exp_result["snr_best_p_ie"] = float(pies_snr[best_snr_idx])
                _plot_snr_three_panels(
                    snrs,
                    pfas_snr,
                    pmds_snr,
                    pies_snr,
                    save_path=os.path.join(fig_dir, f"snr_{exp_name}_rho{rho_db}_k{k_snr}.png"),
                    title=f"{label} | Barrido Eb/No | k={k_snr}, n={n}",
                    show=show_figures,
                )

            if do_tau_sweep and (k_tau < n) and k_is_valid_for_5g(k_tau, n):
                ebno_tau = rho_db_to_ebno_db(float(rho_db), k_tau / n)

                def make_sys_tau(k_: int, n_: int) -> ActivityAwarePolarSystem:
                    return ActivityAwarePolarSystem(k=k_, n=n_, ae_model=ae, p_empty=0.3, thresh=0.5)

                taus, pfas_t, pmds_t, pies_t, _, best_tau = run_threshold_sweep(
                    n=n,
                    k=k_tau,
                    ebno_db=float(ebno_tau),
                    make_system=make_sys_tau,
                    label=label,
                    best_by="p_ie",
                    figure_dir=fig_dir,
                    show_figures=show_figures,
                )
                exp_result["tau_best"] = float(best_tau)
                for i, tau in enumerate(taus):
                    _append_csv(
                        os.path.join(tbl_dir, "tau_sweep.csv"),
                        tau_fields,
                        {
                            "run_id": run_id,
                            "exp_name": exp_name,
                            "rho_db": float(rho_db),
                            "tau": float(tau),
                            "k": int(k_tau),
                            "n": int(n),
                            "R": float(k_tau / n),
                            "ebno_db": float(ebno_tau),
                            "p_fa": float(pfas_t[i]),
                            "p_md": float(pmds_t[i]),
                            "p_ie": float(pies_t[i]),
                        },
                    )

            summary["results"].append(exp_result)
            with open(os.path.join(base, "summary.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

    print(f"\n[OK] Pipeline completo finalizado. Carpeta: {base}")
    return summary


if __name__ == "__main__":
    run_full_pipeline(show_figures=False, use_wandb=False)
