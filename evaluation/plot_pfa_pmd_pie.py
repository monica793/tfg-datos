import numpy as np
import matplotlib.pyplot as plt
from sionna.phy.fec.polar import Polar5GEncoder

from systems.hybrid_polar import ActivityAwarePolarSystem
from utils.signal import rho_db_to_ebno_db


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


def eval_pfa_pmd_pie(n, k, rho_db, ae):
    """
    Evalúa Pfa, Pmd y Pie para tasa R=k/n a un SNR dado.
    Devuelve (R, pfa, pmd, pie).
    """
    R      = k / n
    ebno_db = rho_db_to_ebno_db(rho_db, R)
    system  = ActivityAwarePolarSystem(k=k, n=n, ae_model=ae,
                                       p_empty=P_EMPTY, thresh=THRESH)

    fa = md = n_empty = n_active = inclusive_err = total = 0

    for _ in range(N_BATCHES_EVAL):
        u, u_hat, c_true, c_hat, a_true, p_act, a_hat = system(
            batch_size=BATCH_SIZE_SIM,
            ebno_db=float(ebno_db)
        )

        a_true_b = a_true.squeeze() > 0.5
        a_hat_b  = a_hat.squeeze()  > 0.5

        fa       += int(np.sum(~a_true_b & a_hat_b))
        n_empty  += int(np.sum(~a_true_b))
        md       += int(np.sum(a_true_b & ~a_hat_b))
        n_active += int(np.sum(a_true_b))

        # Pie: error inclusivo por bloque
        blk_err = a_true_b & ~a_hat_b  # activo no detectado

        # Activo y detectado pero con error de decodificación
        mask_ad = a_true_b & a_hat_b
        if np.any(mask_ad):
            u_ad    = u[mask_ad]
            uhat_ad = np.clip(np.round(u_hat[mask_ad]), 0, 1)
            blk_ad_err = np.any(u_ad != uhat_ad, axis=-1)
            tmp = np.zeros(BATCH_SIZE_SIM, dtype=bool)
            tmp[np.where(mask_ad)[0]] = blk_ad_err
            blk_err = blk_err | tmp

        blk_err = blk_err | (~a_true_b & a_hat_b)  # falsa alarma

        inclusive_err += int(np.sum(blk_err))
        total         += BATCH_SIZE_SIM

    return R, fa / max(n_empty, 1), md / max(n_active, 1), inclusive_err / max(total, 1)


def run_curves_for_n(n, rho_db, ae):
    """
    Barre todos los k válidos y genera curvas Pfa/Pmd/Pie vs R=k/n.
    Guarda la figura en results/figures/.
    """
    valid_ks = [k for k in K_CAND if k < n and k_is_valid_for_5g(k, n)]
    if not valid_ks:
        print("[ERROR] No hay k válidos para este n.")
        return

    Rs, PFAs, PMDs, PIEs = [], [], [], []
    print(f"\n=== Evaluación vs k | n={n} | rho={rho_db} dB | thresh={THRESH} ===")

    for k in valid_ks:
        try:
            R, pfa, pmd, pie = eval_pfa_pmd_pie(n, k, rho_db, ae)
            Rs.append(R); PFAs.append(pfa); PMDs.append(pmd); PIEs.append(pie)
            print(f"k={k:3d} | R={R:.3f} | Pfa={pfa:.2e} | Pmd={pmd:.2e} | Pie={pie:.2e}")
        except Exception as e:
            print(f"[skip] k={k} -> {type(e).__name__}: {e}")

    if not Rs:
        print("[ERROR] No se pudo evaluar ningún k.")
        return

    plt.figure(figsize=(9, 5))
    plt.semilogy(Rs, PFAs, marker="o", label="P_FA")
    plt.semilogy(Rs, PMDs, marker="s", label="P_MD")
    plt.semilogy(Rs, PIEs, marker="^", label="P_IE (inclusivo)")
    plt.grid(True, which="both", alpha=0.35)
    plt.xlabel("R = k/n")
    plt.ylabel("Probabilidad")
    plt.title(f"n={n} | rho={rho_db} dB | p_empty={P_EMPTY} | thresh={THRESH}")
    plt.legend()

    import os
    os.makedirs("results/figures", exist_ok=True)
    fname = f"results/figures/curves_n{n}_rho{rho_db}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Figura guardada: {fname}")
    plt.show()


if __name__ == "__main__":
    from training.train_hybrid import train_ae_for_n, k_is_valid_for_5g, N_FIXED, RHO_DBS, K_CAND

    for rho_db in RHO_DBS:
        valid_ks = [k for k in K_CAND if k < N_FIXED and k_is_valid_for_5g(k, N_FIXED)]
        k_train  = valid_ks[len(valid_ks) // 2]
        ae       = train_ae_for_n(n=N_FIXED, rho_db=rho_db, k_train=k_train)
        run_curves_for_n(n=N_FIXED, rho_db=rho_db, ae=ae)