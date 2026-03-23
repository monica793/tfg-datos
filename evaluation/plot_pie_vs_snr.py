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
    Evalúa Pfa, Pmd y Pie para un sistema a un Eb/No concreto.
    system(batch_size, ebno_db) -> u, u_hat, a_true, p_active, a_hat  (numpy)
    """
    fa = md = n_empty = n_active = inclusive_err = total = 0

    for _ in range(N_BATCHES_EVAL):
        outputs = system(batch_size=BATCH_SIZE_SIM, ebno_db=float(ebno_db))

        # Compatible con 5 salidas (e2e) y 7 salidas (híbrido)
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

        blk_err = a_true_b & ~a_hat_b

        mask_ad = a_true_b & a_hat_b
        if np.any(mask_ad):
            u_ad       = u[mask_ad]
            uhat_ad    = np.clip(np.round(u_hat[mask_ad]), 0, 1)
            blk_ad_err = np.any(u_ad != uhat_ad, axis=-1)
            tmp        = np.zeros(BATCH_SIZE_SIM, dtype=bool)
            tmp[np.where(mask_ad)[0]] = blk_ad_err
            blk_err    = blk_err | tmp

        blk_err       = blk_err | (~a_true_b & a_hat_b)
        inclusive_err += int(np.sum(blk_err))
        total         += BATCH_SIZE_SIM

    return (fa  / max(n_empty,  1),
            md  / max(n_active, 1),
            inclusive_err / max(total, 1))


def run_curves_vs_snr(systems: dict, k=K, n=N, ebno_range=None, save=True):
    """
    Genera curvas Pfa/Pmd/Pie vs Eb/No para cada sistema en el dict.

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

    results = {}   # {nombre: (SNRs, PFAs, PMDs, PIEs)}

    for label, make_system in systems.items():
        print(f"\n=== {label} | k={k} | n={n} ===")
        SNRs, PFAs, PMDs, PIEs = [], [], [], []
        system = make_system()

        for ebno_db in ebno_range:
            try:
                pfa, pmd, pie = eval_metrics_at_snr(system, ebno_db)
                SNRs.append(ebno_db)
                PFAs.append(pfa); PMDs.append(pmd); PIEs.append(pie)
                print(f"Eb/No={ebno_db:+3d} dB | Pfa={pfa:.2e} | Pmd={pmd:.2e} | Pie={pie:.2e}")
            except Exception as e:
                print(f"[skip] Eb/No={ebno_db} dB -> {type(e).__name__}: {e}")

        results[label] = (SNRs, PFAs, PMDs, PIEs)

    _plot_comparison(results, k=k, n=n, save=save)
    return results


def _plot_comparison(results: dict, k, n, save=True):
    """Genera la figura comparativa con los tres paneles."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics   = ["P_FA", "P_MD", "P_IE (inclusivo)"]
    idx       = [1, 2, 3]   # PFAs=idx1, PMDs=idx2, PIEs=idx3 dentro de la tupla

    for label, (SNRs, PFAs, PMDs, PIEs) in results.items():
        for ax, vals in zip(axes, [PFAs, PMDs, PIEs]):
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