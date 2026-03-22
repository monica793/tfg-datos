import os
import numpy as np
import matplotlib.pyplot as plt
from sionna.phy.fec.polar import Polar5GEncoder


# ============================================================
# Configuración
# ============================================================
N_FIXED        = 100
P_EMPTY        = 0.30
THRESH         = 0.50
RHO_DBS        = [0.0, 3.0]
K_CAND         = list(range(5, N_FIXED + 1, 5))
BATCH_SIZE_SIM = 3000
N_BATCHES_EVAL = 40


def k_is_valid_for_5g(k, n):
    try:
        Polar5GEncoder(k=int(k), n=int(n))
        return True
    except Exception:
        return False


def eval_pfa_pmd_pie(system, n, k, ebno_db):
    """
    Evalúa Pfa, Pmd y Pie para cualquier sistema con interfaz:
        system(batch_size, ebno_db) -> u, u_hat, a_true, p_active, a_hat
    Devuelve (R, pfa, pmd, pie).
    """
    R = k / n
    fa = md = n_empty = n_active = inclusive_err = total = 0

    for _ in range(N_BATCHES_EVAL):
        # Llamada agnóstica: el sistema devuelve numpy
        outputs = system(batch_size=BATCH_SIZE_SIM, ebno_db=float(ebno_db))

        # Soporte para sistemas con 5 salidas (e2e) o 7 salidas (híbrido)
        if len(outputs) == 5:
            u, u_hat, a_true, p_active, a_hat = outputs
        else:
            u, u_hat, _, _, a_true, p_active, a_hat = outputs

        a_true_b = a_true.squeeze() > 0.5
        a_hat_b  = a_hat.squeeze()  > 0.5

        fa       += int(np.sum(~a_true_b & a_hat_b))
        n_empty  += int(np.sum(~a_true_b))
        md       += int(np.sum(a_true_b & ~a_hat_b))
        n_active += int(np.sum(a_true_b))

        # Pie: error inclusivo por bloque
        blk_err = a_true_b & ~a_hat_b

        mask_ad = a_true_b & a_hat_b
        if np.any(mask_ad):
            u_ad    = u[mask_ad]
            uhat_ad = np.clip(np.round(u_hat[mask_ad]), 0, 1)
            blk_ad_err = np.any(u_ad != uhat_ad, axis=-1)
            tmp = np.zeros(BATCH_SIZE_SIM, dtype=bool)
            tmp[np.where(mask_ad)[0]] = blk_ad_err
            blk_err = blk_err | tmp

        blk_err       = blk_err | (~a_true_b & a_hat_b)
        inclusive_err += int(np.sum(blk_err))
        total         += BATCH_SIZE_SIM

    return R, fa / max(n_empty, 1), md / max(n_active, 1), inclusive_err / max(total, 1)


def run_curves_for_n(n, rho_db, make_system, label='Sistema', k_cand=None):
    """
    Barre k válidos, evalúa y genera curvas Pfa/Pmd/Pie vs R=k/n.
    Guarda la figura en results/figures/.

    make_system: función (k, n) -> sistema instanciado
    label:       nombre del sistema para el título
    """
    from utils.signal import rho_db_to_ebno_db

    if k_cand is None:
        k_cand = K_CAND

    valid_ks = [k for k in k_cand if k < n and k_is_valid_for_5g(k, n)]
    if not valid_ks:
        print("[ERROR] No hay k válidos.")
        return

    Rs, PFAs, PMDs, PIEs = [], [], [], []
    print(f"\n=== Evaluación vs k | {label} | n={n} | rho={rho_db} dB ===")

    for k in valid_ks:
        try:
            ebno_db = rho_db_to_ebno_db(rho_db, k / n)
            system  = make_system(k, n)
            R, pfa, pmd, pie = eval_pfa_pmd_pie(system, n, k, ebno_db)
            Rs.append(R); PFAs.append(pfa); PMDs.append(pmd); PIEs.append(pie)
            print(f"k={k:3d} | R={R:.3f} | Pfa={pfa:.2e} | Pmd={pmd:.2e} | Pie={pie:.2e}")
        except Exception as e:
            print(f"[skip] k={k} -> {type(e).__name__}: {e}")

    if not Rs:
        print("[ERROR] No se pudo evaluar ningún k.")
        return

    _plot_and_save(Rs, PFAs, PMDs, PIEs,
                   title=f"{label} | n={n} | rho={rho_db} dB | p_empty={P_EMPTY}",
                   fname=f"results/figures/{label.replace(' ','_')}_n{n}_rho{rho_db}.png")

    return Rs, PFAs, PMDs, PIEs


def plot_comparison(n, rho_db, systems: dict):
    """
    Genera una gráfica comparativa con múltiples sistemas.

    systems: dict {nombre: make_system_fn}  donde make_system_fn(k,n) -> sistema
    """
    from utils.signal import rho_db_to_ebno_db

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["P_FA", "P_MD", "P_IE (inclusivo)"]

    for label, make_system in systems.items():
        valid_ks = [k for k in K_CAND if k < n and k_is_valid_for_5g(k, n)]
        Rs, PFAs, PMDs, PIEs = [], [], [], []

        for k in valid_ks:
            try:
                ebno_db = rho_db_to_ebno_db(rho_db, k / n)
                system  = make_system(k, n)
                R, pfa, pmd, pie = eval_pfa_pmd_pie(system, n, k, ebno_db)
                Rs.append(R); PFAs.append(pfa); PMDs.append(pmd); PIEs.append(pie)
            except Exception as e:
                print(f"[skip] {label} k={k} -> {e}")

        for ax, vals in zip(axes, [PFAs, PMDs, PIEs]):
            ax.semilogy(Rs, vals, marker='o', label=label)

    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.set_xlabel("R = k/n")
        ax.set_ylabel("Probabilidad")
        ax.grid(True, which="both", alpha=0.35)
        ax.legend()

    fig.suptitle(f"Comparativa | n={n} | rho={rho_db} dB | p_empty={P_EMPTY}")
    plt.tight_layout()

    os.makedirs("results/figures", exist_ok=True)
    fname = f"results/figures/comparison_n{n}_rho{rho_db}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Figura guardada: {fname}")
    plt.show()


def _plot_and_save(Rs, PFAs, PMDs, PIEs, title, fname):
    plt.figure(figsize=(9, 5))
    plt.semilogy(Rs, PFAs, marker="o", label="P_FA")
    plt.semilogy(Rs, PMDs, marker="s", label="P_MD")
    plt.semilogy(Rs, PIEs, marker="^", label="P_IE (inclusivo)")
    plt.grid(True, which="both", alpha=0.35)
    plt.xlabel("R = k/n")
    plt.ylabel("Probabilidad")
    plt.title(title)
    plt.legend()
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Figura guardada: {fname}")
    plt.show()