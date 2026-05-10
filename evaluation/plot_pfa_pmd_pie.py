import os
import numpy as np
import matplotlib.pyplot as plt

from utils.representative_ks import pick_representative_ks


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
    """
    k admisible para Polar5G(n). Solo se usa con polar_k_constraint=True (híbrido).
    Importa Sionna de forma diferida para no cargarlo en rutas solo E2E.
    """
    try:
        from sionna.phy.fec.polar import Polar5GEncoder

        Polar5GEncoder(k=int(k), n=int(n))
        return True
    except Exception:
        return False


def _binomial_ci_95(num, den):
    """
    Intervalo de confianza 95% (aprox. normal/Wald) para una proporción.
    Devuelve (low, high).
    """
    den = int(den)
    num = int(num)
    if den <= 0:
        return np.nan, np.nan
    p = num / den
    margin = 1.96 * np.sqrt(max(p * (1.0 - p), 0.0) / den)
    return max(0.0, p - margin), min(1.0, p + margin)


def _safe_ratio(num, den):
    den = int(den)
    if den <= 0:
        return np.nan
    return float(num) / float(den)


def eval_pfa_pmd_pie(system, n, k, ebno_db, return_ci=False, thresh_override=None):
    """
    Evalúa Pfa, Pmd, P_IE (Lancho) y P_global para cualquier sistema con interfaz:
        system(batch_size, ebno_db) -> u, u_hat, a_true, p_active, a_hat
    Devuelve:
      - Si return_ci=False: (R, p_fa, p_md, p_ie, p_global)
      - Si return_ci=True :
        (R, p_fa, p_md, p_ie, p_global, p_fa_ci, p_md_ci, p_ie_ci, p_global_ci)
    """
    R = k / n
    false_alarms = missed_detections = 0
    n_empty = n_active = 0
    active_detected_decoding_errors = 0
    total = 0
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

            empty_mask = ~a_true_b
            active_mask = a_true_b
            md_mask = active_mask & ~a_hat_b
            mask_ad = a_true_b & a_hat_b

            false_alarms += int(np.sum(empty_mask & a_hat_b))
            n_empty += int(np.sum(empty_mask))
            missed_detections += int(np.sum(md_mask))
            n_active += int(np.sum(active_mask))

            if np.any(mask_ad):
                u_ad    = u[mask_ad]
                uhat_ad = np.clip(np.round(u_hat[mask_ad]), 0, 1)
                blk_ad_err = np.any(u_ad != uhat_ad, axis=-1)
                active_detected_decoding_errors += int(np.sum(blk_ad_err))

            total += int(a_true_b.shape[0])
    finally:
        if old_thresh is not None:
            system.thresh = old_thresh

    inclusive_errors_active = missed_detections + active_detected_decoding_errors
    global_errors = inclusive_errors_active + false_alarms

    p_fa = _safe_ratio(false_alarms, n_empty)
    p_md = _safe_ratio(missed_detections, n_active)
    p_ie = _safe_ratio(inclusive_errors_active, n_active)
    p_global = _safe_ratio(global_errors, total)

    if n_active > 0:
        assert np.isclose(
            p_ie,
            (missed_detections + active_detected_decoding_errors) / n_active
        )
    if n_empty > 0:
        assert np.isclose(p_fa, false_alarms / n_empty)
    if total > 0:
        assert np.isclose(p_global, global_errors / total)

    if not return_ci:
        return R, p_fa, p_md, p_ie, p_global

    p_fa_ci = _binomial_ci_95(false_alarms, n_empty)
    p_md_ci = _binomial_ci_95(missed_detections, n_active)
    p_ie_ci = _binomial_ci_95(inclusive_errors_active, n_active)
    p_global_ci = _binomial_ci_95(global_errors, total)
    return R, p_fa, p_md, p_ie, p_global, p_fa_ci, p_md_ci, p_ie_ci, p_global_ci


def run_curves_for_n(
    n,
    rho_db,
    make_system,
    label='Sistema',
    k_cand=None,
    figure_path=None,
    show_figure=True,
    polar_k_constraint=True,
):
    """
    Barre k válidos, evalúa y genera curvas Pfa/Pmd/P_IE/P_global vs R=k/n.
    Guarda la figura en results/figures/ o en figure_path si se indica.

    make_system: función (k, n) -> sistema instanciado
    label:       nombre del sistema para el título
    polar_k_constraint: si True, solo k admitidos por Polar5G(n) (como el híbrido).
        False para E2E u otros sistemas donde cualquier k < n es válido.
    """
    from utils.signal import rho_db_to_ebno_db

    if k_cand is None:
        k_cand = K_CAND

    if polar_k_constraint:
        valid_ks = [k for k in k_cand if k < n and k_is_valid_for_5g(k, n)]
    else:
        valid_ks = [k for k in k_cand if k < n]
    if not valid_ks:
        print("[ERROR] No hay k válidos.")
        return

    Rs, PFAs, PMDs, PIEs, P_GLOBALs = [], [], [], [], []
    PFA_CIs, PMD_CIs, PIE_CIs, P_GLOBAL_CIs = [], [], [], []
    print(f"\n=== Evaluación vs k | {label} | n={n} | rho={rho_db} dB ===")

    for k in valid_ks:
        try:
            ebno_db = rho_db_to_ebno_db(rho_db, k / n)
            system  = make_system(k, n)
            R, p_fa, p_md, p_ie, p_global, p_fa_ci, p_md_ci, p_ie_ci, p_global_ci = eval_pfa_pmd_pie(
                system, n, k, ebno_db, return_ci=True
            )
            Rs.append(R); PFAs.append(p_fa); PMDs.append(p_md); PIEs.append(p_ie); P_GLOBALs.append(p_global)
            PFA_CIs.append(p_fa_ci); PMD_CIs.append(p_md_ci); PIE_CIs.append(p_ie_ci); P_GLOBAL_CIs.append(p_global_ci)
            print(
                f"k={k:3d} | R={R:.3f} | "
                f"P_FA={p_fa:.2e} [{p_fa_ci[0]:.2e}, {p_fa_ci[1]:.2e}] | "
                f"P_MD={p_md:.2e} [{p_md_ci[0]:.2e}, {p_md_ci[1]:.2e}] | "
                f"P_IE={p_ie:.2e} [{p_ie_ci[0]:.2e}, {p_ie_ci[1]:.2e}] | "
                f"P_global={p_global:.2e} [{p_global_ci[0]:.2e}, {p_global_ci[1]:.2e}]"
            )
        except Exception as e:
            print(f"[skip] k={k} -> {type(e).__name__}: {e}")

    if not Rs:
        print("[ERROR] No se pudo evaluar ningún k.")
        return

    if figure_path is None:
        figure_path = f"results/figures/{label.replace(' ','_')}_n{n}_rho{rho_db}.png"
    _plot_and_save(
        Rs, PFAs, PMDs, PIEs, P_GLOBALs,
        title=f"{label} | n={n} | rho={rho_db} dB | p_empty={P_EMPTY}",
        fname=figure_path,
        pfa_cis=PFA_CIs,
        pmd_cis=PMD_CIs,
        pie_cis=PIE_CIs,
        p_global_cis=P_GLOBAL_CIs,
        show=show_figure,
    )

    return Rs, PFAs, PMDs, PIEs, P_GLOBALs


def plot_comparison(n, rho_db, systems: dict, k_cand=None, polar_k_constraint=True):
    """
    Genera una gráfica comparativa con múltiples sistemas.

    systems: dict {nombre: make_system_fn}  donde make_system_fn(k,n) -> sistema
    k_cand: lista de k a barrer (por defecto K_CAND completo).
    polar_k_constraint: igual que en run_curves_for_n.
    """
    from utils.signal import rho_db_to_ebno_db

    if k_cand is None:
        k_cand = K_CAND

    fig, axes = plt.subplots(1, 4, figsize=(19, 5))
    titles = ["P_FA", "P_MD", "P_IE", "P_global"]

    for label, make_system in systems.items():
        if polar_k_constraint:
            valid_ks = [k for k in k_cand if k < n and k_is_valid_for_5g(k, n)]
        else:
            valid_ks = [k for k in k_cand if k < n]
        Rs, PFAs, PMDs, PIEs, P_GLOBALs = [], [], [], [], []

        for k in valid_ks:
            try:
                ebno_db = rho_db_to_ebno_db(rho_db, k / n)
                system  = make_system(k, n)
                R, p_fa, p_md, p_ie, p_global = eval_pfa_pmd_pie(system, n, k, ebno_db)
                Rs.append(R); PFAs.append(p_fa); PMDs.append(p_md); PIEs.append(p_ie); P_GLOBALs.append(p_global)
            except Exception as e:
                print(f"[skip] {label} k={k} -> {e}")

        for ax, vals in zip(axes, [PFAs, PMDs, PIEs, P_GLOBALs]):
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


def _plot_and_save(Rs, PFAs, PMDs, PIEs, P_GLOBALs, title, fname,
                   pfa_cis=None, pmd_cis=None, pie_cis=None, p_global_cis=None,
                   show=True):
    plt.figure(figsize=(9, 5))
    for vals, cis, marker, label in [
        (PFAs, pfa_cis, "o", "P_FA"),
        (PMDs, pmd_cis, "s", "P_MD"),
        (PIEs, pie_cis, "^", "P_IE"),
        (P_GLOBALs, p_global_cis, "d", "P_global"),
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
    os.makedirs(os.path.dirname(fname) or ".", exist_ok=True)
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Figura guardada: {fname}")
    if show:
        plt.show()
    else:
        plt.close()


def run_threshold_sweep(
    n,
    k,
    ebno_db,
    make_system,
    label="Sistema",
    thresholds=None,
    best_by="p_ie",
    figure_dir=None,
    show_figures=True,
):
    """
    Barre umbral tau de actividad y genera:
      - ROC (P_MD vs P_FA)
      - P_IE y P_global vs tau

    Nota: se reevalúa el sistema para cada tau para respetar la lógica
    de decodificación condicionada por actividad.
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    thresholds = [float(t) for t in thresholds]

    if best_by not in {"p_ie", "p_global"}:
        raise ValueError("best_by debe ser 'p_ie' o 'p_global'")

    PFAs, PMDs, PIEs, P_GLOBALs = [], [], [], []
    print(f"\n=== Barrido de umbral | {label} | n={n} | k={k} | Eb/No={ebno_db:.2f} dB ===")

    for tau in thresholds:
        system = make_system(k, n)
        _, p_fa, p_md, p_ie, p_global = eval_pfa_pmd_pie(system, n, k, ebno_db, thresh_override=tau)
        PFAs.append(p_fa); PMDs.append(p_md); PIEs.append(p_ie); P_GLOBALs.append(p_global)
        print(f"tau={tau:.2f} | P_FA={p_fa:.2e} | P_MD={p_md:.2e} | P_IE={p_ie:.2e} | P_global={p_global:.2e}")

    metric_for_best = PIEs if best_by == "p_ie" else P_GLOBALs
    best_idx = int(np.nanargmin(np.array(metric_for_best, dtype=float)))
    best_tau = thresholds[best_idx]
    best_metric = metric_for_best[best_idx]
    print(f"[BEST] criterio={best_by} | tau*={best_tau:.3f} -> {best_by}={best_metric:.2e}")

    if figure_dir is None:
        figure_dir = "results/figures"
    os.makedirs(figure_dir, exist_ok=True)
    safe_label = label.replace(' ', '_')

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
    fname_roc = os.path.join(figure_dir, f"roc_{safe_label}_n{n}_k{k}_ebno{ebno_db}.png")
    plt.savefig(fname_roc, dpi=150, bbox_inches="tight")
    print(f"Figura guardada: {fname_roc}")
    if show_figures:
        plt.show()
    else:
        plt.close()

    # P_IE y P_global vs tau
    plt.figure(figsize=(6.5, 5))
    plt.semilogy(thresholds, PIEs, marker="o", label="P_IE")
    plt.semilogy(thresholds, P_GLOBALs, marker="s", label="P_global")
    plt.axvline(best_tau, color="r", linestyle="--", label=f"tau*={best_tau:.2f}")
    plt.grid(True, which="both", alpha=0.35)
    plt.xlabel("Umbral tau")
    plt.ylabel("Probabilidad")
    plt.title(f"P_IE/P_global vs tau | {label} | n={n} | k={k} | Eb/No={ebno_db:.2f} dB")
    plt.legend()
    fname_pie = os.path.join(figure_dir, f"pie_pglobal_vs_tau_{safe_label}_n{n}_k{k}_ebno{ebno_db}.png")
    plt.savefig(fname_pie, dpi=150, bbox_inches="tight")
    print(f"Figura guardada: {fname_pie}")
    if show_figures:
        plt.show()
    else:
        plt.close()

    return thresholds, PFAs, PMDs, PIEs, P_GLOBALs, best_tau