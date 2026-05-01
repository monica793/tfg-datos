"""
Runner de experimentos híbridos.

Cada experimento define solo lo que cambia respecto a los valores por defecto.
El resto de parámetros se hereda de train_hybrid.py.
"""
import json
import os
from datetime import datetime

from evaluation.plot_pfa_pmd_pie import run_curves_for_n
from systems.hybrid_polar import ActivityAwarePolarSystem
from training.train_hybrid import K_CAND, N_FIXED, RHO_DBS, k_is_valid_for_5g, train_ae_for_n

# ============================================================
# Experimentos definidos como lista de dicts
# Solo se especifica lo que cambia respecto al default
# ============================================================
EXPERIMENTS = [
    {"name": "base",         "w_recon": 1.0, "w_class": 10.0, "latent_dim": 64, "hidden_dim": 256},
    {"name": "recon_fuerte", "w_recon": 3.0, "w_class": 10.0, "latent_dim": 64, "hidden_dim": 256},
    {"name": "class_fuerte", "w_recon": 1.0, "w_class": 20.0, "latent_dim": 64, "hidden_dim": 256},
    {"name": "arq_grande",   "w_recon": 1.0, "w_class": 10.0, "latent_dim": 96, "hidden_dim": 384},
]


def run_all(experiments=None, n=N_FIXED, rho_dbs=None, ae_epochs=20,
            ae_steps=300, use_wandb=True):
    """
    Entrena y evalúa cada experimento secuencialmente.
    Guarda checkpoints y curvas por separado para cada config.
    Devuelve un resumen con todos los resultados.
    """
    if experiments is None:
        experiments = EXPERIMENTS
    if rho_dbs is None:
        rho_dbs = RHO_DBS

    valid_ks = [k for k in K_CAND if k < n and k_is_valid_for_5g(k, n) and not (12 <= k <= 19)]
    os.makedirs("results/experiments", exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {"run_id": run_id, "n": n, "valid_ks": valid_ks, "experiments": []}
    out_json = f"results/experiments/summary_{run_id}.json"

    for exp in experiments:
        name = exp["name"]
        print(f"\n{'='*70}\n[EXP] {name}\n{'='*70}")

        trained = {}
        for rho_db in rho_dbs:
            trained[rho_db] = train_ae_for_n(
                n=n, rho_db=rho_db, valid_ks=valid_ks,
                use_wandb=use_wandb, checkpoint_tag=name,
                latent_dim=exp.get("latent_dim", 64),
                hidden_dim=exp.get("hidden_dim", 256),
                dropout=exp.get("dropout", 0.1),
                ae_epochs=ae_epochs,
                ae_steps_per_epoch=ae_steps,
                w_recon=exp.get("w_recon", 1.0),
                w_class=exp.get("w_class", 10.0),
            )

        exp_record = {"name": name, "config": exp, "results": {}}
        for rho_db, ae in trained.items():
            def make_system(k, n_):
                return ActivityAwarePolarSystem(k=k, n=n_, ae_model=ae, p_empty=0.3, thresh=0.5)

            out = run_curves_for_n(n=n, rho_db=rho_db, make_system=make_system,
                                   label=f"Hibrido_{name}", k_cand=valid_ks)
            if out:
                Rs, PFAs, PMDs, PIEs = out
                best = int(min(range(len(PIEs)), key=lambda i: PIEs[i]))
                exp_record["results"][str(rho_db)] = {
                    "best_pie": float(PIEs[best]),
                    "best_R": float(Rs[best]),
                }

        summary["experiments"].append(exp_record)
        with open(out_json, "w") as f:
            json.dump(summary, f, indent=2)

    print(f"\n[OK] Resumen guardado: {out_json}")
    return summary


if __name__ == "__main__":
    run_all()
