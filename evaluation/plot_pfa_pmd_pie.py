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


def _binomial_ci_95(num, den):
    """
    Intervalo de confianza 95% (aprox. normal/Wald) para una proporción.
    Devuelve (low, high).
    """
    den = int(den)
    num = int(num)
    if den <= 0:
        return 0.0, 0.0
    p = num / den
    margin = 1.96 * np.sqrt(max(p * (1.0 - p), 0.0) / den)
    return max(0.0, p - margin), min(1.0, p + margin)


def k_is_valid_for_5g(k, n):
    try:
        Polar5GEncoder(k=int(k), n=int(n))
        return True
    except Exception:
        return False


def eval_pfa_pmd_pie(system, n, k, ebno_db, return_ci=False, thresh_override=None):
    """
    Evalúa Pfa, Pmd y Pie para cualquier sistema con interfaz:
        system(batch_size, ebno_db) -> u, u_hat, a_true, p_active, a_hat
    Devuelve:
      - Si return_ci=False: (R, pfa, pmd, pie)
      - Si return_ci=True : (R, pfa, pmd, pie, pfa_ci, pmd_ci, pie_ci)
    """
    R = k / n
    fa = md = n_empty = n_active = inclusive_err = total = 0
    old_thresh = None

    if thresh_override is not None and hasattr(system, "thresh"):
        old_thresh = float(system.thresh)
        system.thresh = float(thresh_override)

    try:
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
    finally:
        if old_thresh is not None:
            system.thresh = old_thresh

    pfa = fa / max(n_empty, 1)
    pmd = md / max(n_active, 1)
    pie = inclusive_err / max(total, 1)

    if not return_ci:
        return R, pfa, pmd, pie

    pfa_ci = _binomial_ci_95(fa, n_empty)
    pmd_ci = _binomial_ci_95(md, n_active)
    pie_ci = _binomial_ci_95(inclusive_err, total)
    return R, pfa, pmd, pie, pfa_ci, pmd_ci, pie_ci


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
    PFA_CIs, PMD_CIs, PIE_CIs = [], [], []
    print(f"\n=== Evaluación vs k | {label} | n={n} | rho={rho_db} dB ===")

    for k in valid_ks:
        try:
            ebno_db = rho_db_to_ebno_db(rho_db, k / n)
            system  = make_system(k, n)
            R, pfa, pmd, pie, pfa_ci, pmd_ci, pie_ci = eval_pfa_pmd_pie(
                system, n, k, ebno_db, return_ci=True
            )
            Rs.append(R); PFAs.append(pfa); PMDs.append(pmd); PIEs.append(pie)
            PFA_CIs.append(pfa_ci); PMD_CIs.append(pmd_ci); PIE_CIs.append(pie_ci)
            print(
                f"k={k:3d} | R={R:.3f} | "
                f"Pfa={pfa:.2e} [{pfa_ci[0]:.2e}, {pfa_ci[1]:.2e}] | "
                f"Pmd={pmd:.2e} [{pmd_ci[0]:.2e}, {pmd_ci[1]:.2e}] | "
                f"Pie={pie:.2e} [{pie_ci[0]:.2e}, {pie_ci[1]:.2e}]"
            )
        except Exception as e:
            print(f"[skip] k={k} -> {type(e).__name__}: {e}")

    if not Rs:
        print("[ERROR] No se pudo evaluar ningún k.")
        return

    _plot_and_save(Rs, PFAs, PMDs, PIEs,
                   title=f"{label} | n={n} | rho={rho_db} dB | p_empty={P_EMPTY}",
                   fname=f"results/figures/{label.replace(' ','_')}_n{n}_rho{rho_db}.png",
                   pfa_cis=PFA_CIs, pmd_cis=PMD_CIs, pie_cis=PIE_CIs)

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


def _plot_and_save(Rs, PFAs, PMDs, PIEs, title, fname,
                   pfa_cis=None, pmd_cis=None, pie_cis=None):
    plt.figure(figsize=(9, 5))
    for vals, cis, marker, label in [
        (PFAs, pfa_cis, "o", "P_FA"),
        (PMDs, pmd_cis, "s", "P_MD"),
        (PIEs, pie_cis, "^", "P_IE (inclusivo)"),
    ]:
        if cis is not None and len(cis) == len(vals):
            low = np.array([c[0] for c in cis], dtype=float)
            high = np.array([c[1] for c in cis], dtype=float)
            y = np.array(vals, dtype=float)
            yerr = np.vstack([np.maximum(y - low, 0.0), np.maximum(high - y, 0.0)])
            plt.errorbar(Rs, y, yerr=yerr, marker=marker, linestyle='-', capsize=3, label=label)
        else:
            plt.semilogy(Rs, vals, marker=marker, label=label)

    plt.yscale("log")
    plt.grid(True, which="both", alpha=0.35)
    plt.xlabel("R = k/n")
    plt.ylabel("Probabilidad")
    plt.title(title)
    plt.legend()
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Figura guardada: {fname}")
    plt.show()


def run_threshold_sweep(n, k, ebno_db, make_system, label="Sistema", thresholds=None):
    """
    Barre umbral tau de actividad y genera:
      - ROC (P_MD vs P_FA)
      - P_IE vs tau

    Nota: se reevalúa el sistema para cada tau para respetar la lógica
    de decodificación condicionada por actividad.
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    thresholds = [float(t) for t in thresholds]

    PFAs, PMDs, PIEs = [], [], []
    print(f"\n=== Barrido de umbral | {label} | n={n} | k={k} | Eb/No={ebno_db:.2f} dB ===")

    for tau in thresholds:
        system = make_system(k, n)
        _, pfa, pmd, pie = eval_pfa_pmd_pie(system, n, k, ebno_db, thresh_override=tau)
        PFAs.append(pfa); PMDs.append(pmd); PIEs.append(pie)
        print(f"tau={tau:.2f} | Pfa={pfa:.2e} | Pmd={pmd:.2e} | Pie={pie:.2e}")

    best_idx = int(np.argmin(PIEs))
    best_tau = thresholds[best_idx]
    best_pie = PIEs[best_idx]
    print(f"[BEST] tau*={best_tau:.3f} -> P_IE={best_pie:.2e}")

    os.makedirs("results/figures", exist_ok=True)

    # ROC: eje x P_FA, eje y P_MD
    plt.figure(figsize=(6.5, 5))
    plt.plot(PFAs, PMDs, marker="o")
    for i in [0, len(thresholds)//2, len(thresholds)-1]:
        plt.annotate(f"tau={thresholds[i]:.2f}", (PFAs[i], PMDs[i]))
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", alpha=0.35)
    plt.xlabel("P_FA")
    plt.ylabel("P_MD")
    plt.title(f"ROC | {label} | n={n} | k={k} | Eb/No={ebno_db:.2f} dB")
    fname_roc = f"results/figures/roc_{label.replace(' ','_')}_n{n}_k{k}_ebno{ebno_db}.png"
    plt.savefig(fname_roc, dpi=150, bbox_inches="tight")
    print(f"Figura guardada: {fname_roc}")
    plt.show()

    # P_IE vs tau
    plt.figure(figsize=(6.5, 5))
    plt.semilogy(thresholds, PIEs, marker="o", label="P_IE")
    plt.axvline(best_tau, color="r", linestyle="--", label=f"tau*={best_tau:.2f}")
    plt.grid(True, which="both", alpha=0.35)
    plt.xlabel("Umbral tau")
    plt.ylabel("P_IE")
    plt.title(f"P_IE vs tau | {label} | n={n} | k={k} | Eb/No={ebno_db:.2f} dB")
    plt.legend()
    fname_pie = f"results/figures/pie_vs_tau_{label.replace(' ','_')}_n{n}_k{k}_ebno{ebno_db}.png"
    plt.savefig(fname_pie, dpi=150, bbox_inches="tight")
    print(f"Figura guardada: {fname_pie}")
    plt.show()

    return thresholds, PFAs, PMDs, PIEs, best_tau