import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Configuración
# ============================================================
K               = 50
N               = 100
P_EMPTY         = 0.30
THRESH          = 0.50
BATCH_SIZE_SIM  = 3000
N_BATCHES_EVAL  = 40

# Rango de Eb/No a evaluar (en dB)
EBNO_DB_RANGE   = list(range(-2, 12, 1))   # de -2 a 11 dB


def eval_metrics_at_snr(system, ebno_db):
    """
    Evalúa P_FA, P_MD, P_IE (Lancho) y P_global para un sistema a un Eb/No concreto.
    system(batch_size, ebno_db) -> u, u_hat, a_true, p_active, a_hat  (numpy)
    """
    false_alarms = missed_detections = 0
    n_empty = n_active = 0
    active_detected_decoding_errors = 0
    total = 0

    for _ in range(N_BATCHES_EVAL):
        outputs = system(batch_size=BATCH_SIZE_SIM, ebno_db=float(ebno_db))

        # Compatible con 5 salidas (e2e) y 7 salidas (híbrido)
        if len(outputs) == 5:
            u, u_hat, a_true, p_active, a_hat = outputs
        else:
            u, u_hat, _, _, a_true, p_active, a_hat = outputs

        a_true_b = a_true.squeeze() > 0.5
        a_hat_b  = a_hat.squeeze()  > 0.5

        empty_mask = ~a_true_b
        active_mask = a_true_b
        md_mask = active_mask & ~a_hat_b
        false_alarms += int(np.sum(empty_mask & a_hat_b))
        n_empty += int(np.sum(empty_mask))
        missed_detections += int(np.sum(md_mask))
        n_active += int(np.sum(active_mask))

        mask_ad = a_true_b & a_hat_b
        if np.any(mask_ad):
            u_ad       = u[mask_ad]
            uhat_ad    = np.clip(np.round(u_hat[mask_ad]), 0, 1)
            blk_ad_err = np.any(u_ad != uhat_ad, axis=-1)
            active_detected_decoding_errors += int(np.sum(blk_ad_err))

        total += int(a_true_b.shape[0])

    inclusive_errors_active = missed_detections + active_detected_decoding_errors
    global_errors = inclusive_errors_active + false_alarms

    p_fa = np.nan if n_empty <= 0 else false_alarms / n_empty
    p_md = np.nan if n_active <= 0 else missed_detections / n_active
    p_ie = np.nan if n_active <= 0 else inclusive_errors_active / n_active
    p_global = np.nan if total <= 0 else global_errors / total

    if n_active > 0:
        assert np.isclose(p_ie, inclusive_errors_active / n_active)
    if n_empty > 0:
        assert np.isclose(p_fa, false_alarms / n_empty)
    if total > 0:
        assert np.isclose(p_global, global_errors / total)

    return p_fa, p_md, p_ie, p_global


def run_curves_vs_snr(systems: dict, k=K, n=N, ebno_range=None, save=True):
    """
    Genera curvas P_FA/P_MD/P_IE/P_global vs Eb/No para cada sistema en el dict.

    systems: dict {nombre: make_system_fn}
             make_system_fn() -> sistema instanciado (sin argumentos,
             ya tiene k y n fijos dentro)

    Ejemplo:
        systems = {
            'Híbrido':      lambda: ActivityAwarePolarSystem(k=50, n=100, ae_model=ae),
            'End-to-End':   lambda: E2ESystem(k=50, n=100, encoder=enc, decoder=dec),
        }
    """
    if ebno_range is None:
        ebno_range = EBNO_DB_RANGE

    results = {}   # {nombre: (SNRs, PFAs, PMDs, PIEs, P_GLOBALs)}

    for label, make_system in systems.items():
        print(f"\n=== {label} | k={k} | n={n} ===")
        SNRs, PFAs, PMDs, PIEs, P_GLOBALs = [], [], [], [], []
        system = make_system()

        for ebno_db in ebno_range:
            try:
                p_fa, p_md, p_ie, p_global = eval_metrics_at_snr(system, ebno_db)
                SNRs.append(ebno_db)
                PFAs.append(p_fa); PMDs.append(p_md); PIEs.append(p_ie); P_GLOBALs.append(p_global)
                print(
                    f"Eb/No={ebno_db:+3d} dB | P_FA={p_fa:.2e} | P_MD={p_md:.2e} | "
                    f"P_IE={p_ie:.2e} | P_global={p_global:.2e}"
                )
            except Exception as e:
                print(f"[skip] Eb/No={ebno_db} dB -> {type(e).__name__}: {e}")

        results[label] = (SNRs, PFAs, PMDs, PIEs, P_GLOBALs)

    _plot_comparison(results, k=k, n=n, save=save)
    return results


def _plot_comparison(results: dict, k, n, save=True):
    """Genera la figura comparativa con cuatro paneles."""
    fig, axes = plt.subplots(1, 4, figsize=(19, 5))
    metrics   = ["P_FA", "P_MD", "P_IE", "P_global"]

    for label, (SNRs, PFAs, PMDs, PIEs, P_GLOBALs) in results.items():
        for ax, vals in zip(axes, [PFAs, PMDs, PIEs, P_GLOBALs]):
            ax.semilogy(SNRs, vals, marker='o', label=label)

    for ax, title in zip(axes, metrics):
        ax.set_title(title)
        ax.set_xlabel("Eb/No (dB)")
        ax.set_ylabel("Probabilidad")
        ax.grid(True, which="both", alpha=0.35)
        ax.legend()

    fig.suptitle(f"Comparativa vs SNR | k={k} | n={n} | p_empty={P_EMPTY} | thresh={THRESH}")
    plt.tight_layout()

    if save:
        os.makedirs("results/figures", exist_ok=True)
        fname = f"results/figures/comparison_vs_snr_k{k}_n{n}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"\nFigura guardada: {fname}")

    plt.show()