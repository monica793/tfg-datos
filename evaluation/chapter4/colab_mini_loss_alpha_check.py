"""
Mini cribado Colab: 3 pesos de pérdida × 2 alpha_mix en un solo punto (k, rho).

No guarda checkpoints en Drive. Entrena en RAM (~pocos minutos en GPU).
Salida: tabla con 6 filas (3 entrenamientos, 6 evaluaciones).

Uso en Colab (una celda, tras clonar el repo e instalar deps):

    import os, sys
    os.chdir("/content/tfg-datos")
    sys.path.insert(0, "/content/tfg-datos")
    from evaluation.chapter4.colab_mini_loss_alpha_check import run_mini_screening
    run_mini_screening()
"""

from __future__ import annotations

import time

import numpy as np
import tensorflow as tf
from sionna.phy.fec.polar import Polar5GEncoder
from sionna.phy.mapping import BinarySource, Constellation, Mapper

from models.supervised_ae import SupervisedAE
from systems.hybrid_polar import ActivityAwarePolarSystem
from training.train_hybrid import P_EMPTY, SEED, make_batch
from utils.signal import rho_db_to_ebno_db

# --- Punto fijo (R = k/n = 0.4) ---
N = 100
K = 40
RHO_DB = 0.0
TAU = 0.5

# --- Entrenamiento rápido (subir si quieres más estabilidad) ---
MINI_EPOCHS = 3
MINI_STEPS = 30
MINI_BATCH = 128
LR = 2e-3

# --- Monte Carlo ligero en evaluación ---
MINI_N_BATCHES = 8
MINI_BATCH_SIM = 1500

# (nombre, w_recon, w_class)
WEIGHT_CONFIGS = [
    ("cls_1_10", 1.0, 10.0),
    ("recon_10_1", 10.0, 1.0),
    ("equal_1_1", 1.0, 1.0),
]
ALPHAS = (0.0, 1.0)


def _set_seeds() -> None:
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def _train_ae_single_k(
    *,
    n: int,
    k: int,
    rho_db: float,
    w_recon: float,
    w_class: float,
    epochs: int = MINI_EPOCHS,
    steps_per_epoch: int = MINI_STEPS,
    batch_size: int = MINI_BATCH,
) -> SupervisedAE:
    """Entrena un AE solo en k fijo; pesos solo en memoria."""
    _set_seeds()
    rate = k / n
    ebno_db = rho_db_to_ebno_db(rho_db, rate)

    ae = SupervisedAE(n=n)
    dummy = tf.zeros([1, n, 2])
    ae(dummy, training=False)

    src = BinarySource()
    const = Constellation("pam", num_bits_per_symbol=1, trainable=False)
    enc = Polar5GEncoder(k=k, n=n)
    mapper = Mapper(constellation=const)

    bce = tf.keras.losses.BinaryCrossentropy()
    mse = tf.keras.losses.MeanSquaredError()
    opt = tf.keras.optimizers.Adam(learning_rate=LR)

    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        loss_ep = 0.0
        for _ in range(steps_per_epoch):
            y_ri, x_ri, a = make_batch(src, enc, mapper, ebno_db, rate, batch_size)
            with tf.GradientTape() as tape:
                x_hat_ri, p_active = ae(y_ri, training=True)
                loss = w_recon * mse(x_ri, x_hat_ri) + w_class * bce(a, p_active)
            opt.apply_gradients(zip(tape.gradient(loss, ae.trainable_variables), ae.trainable_variables))
            loss_ep += float(loss.numpy())
        print(f"    época {ep}/{epochs} | loss={loss_ep / steps_per_epoch:.4f}")

    ae.trainable = False
    print(f"    entrenamiento: {time.perf_counter() - t0:.1f} s")
    return ae


def _eval_metrics(
    ae: SupervisedAE,
    *,
    alpha_mix: float,
    n_batches: int = MINI_N_BATCHES,
    batch_size: int = MINI_BATCH_SIM,
) -> tuple[float, float, float]:
    """P_FA, P_MD, P_IE con Monte Carlo reducido."""
    rate = K / N
    ebno_db = rho_db_to_ebno_db(RHO_DB, rate)
    system = ActivityAwarePolarSystem(
        k=K, n=N, ae_model=ae, p_empty=P_EMPTY, thresh=TAU, alpha_mix=alpha_mix
    )

    false_alarms = missed = n_empty = n_active = dec_err = 0

    for _ in range(n_batches):
        outputs = system(batch_size=batch_size, ebno_db=ebno_db)
        u, u_hat, _, _, a_true, _, a_hat = outputs
        a_true_b = a_true.squeeze() > 0.5
        a_hat_b = a_hat.squeeze() > 0.5
        empty = ~a_true_b
        active = a_true_b
        md = active & ~a_hat_b
        ad = active & a_hat_b

        false_alarms += int(np.sum(empty & a_hat_b))
        n_empty += int(np.sum(empty))
        missed += int(np.sum(md))
        n_active += int(np.sum(active))
        if np.any(ad):
            dec_err += int(np.sum(np.any(u[ad] != np.clip(np.round(u_hat[ad]), 0, 1), axis=-1)))

    p_fa = false_alarms / n_empty if n_empty else float("nan")
    p_md = missed / n_active if n_active else float("nan")
    p_ie = (missed + dec_err) / n_active if n_active else float("nan")
    return p_fa, p_md, p_ie


def run_mini_screening(
    *,
    n: int = N,
    k: int = K,
    rho_db: float = RHO_DB,
    verbose: bool = True,
) -> list[dict]:
    """
    3 entrenamientos + 6 evaluaciones. Devuelve lista de dicts con resultados.
    """
    Polar5GEncoder(k=k, n=n)  # falla pronto si k no válido

    print("=" * 72)
    print("MINI CRIBADO: pesos de pérdida × alpha_mix")
    print(f"  n={n}, k={k}, R={k/n:.2f}, rho={rho_db} dB, P_empty={P_EMPTY}, tau={TAU}")
    print(f"  train: {MINI_EPOCHS} épocas × {MINI_STEPS} pasos, batch={MINI_BATCH}")
    print(f"  eval:  {MINI_N_BATCHES} batches × {MINI_BATCH_SIM} ventanas  (~{MINI_N_BATCHES * MINI_BATCH_SIM})")
    print("  NOTA: son 3 redes (una por pesos), NO 6. alpha se cambia solo en evaluación.")
    print("=" * 72)

    models: dict[str, SupervisedAE] = {}
    for name, w_rec, w_cls in WEIGHT_CONFIGS:
        print(f"\n[TRAIN] {name}  (w_recon={w_rec}, w_class={w_cls})")
        models[name] = _train_ae_single_k(
            n=n, k=k, rho_db=rho_db, w_recon=w_rec, w_class=w_cls
        )

    rows: list[dict] = []
    print("\n" + "=" * 72)
    print("[EVAL]")
    header = f"{'pesos':<12} {'w_rec':>5} {'w_cls':>5} {'alpha':>5} {'P_FA':>10} {'P_MD':>10} {'P_IE':>10}  winner?"
    print(header)
    print("-" * len(header))

    for name, w_rec, w_cls in WEIGHT_CONFIGS:
        pies: dict[float, float] = {}
        for alpha in ALPHAS:
            t0 = time.perf_counter()
            p_fa, p_md, p_ie = _eval_metrics(models[name], alpha_mix=alpha)
            dt = time.perf_counter() - t0
            pies[alpha] = p_ie
            row = {
                "weights": name,
                "w_recon": w_rec,
                "w_class": w_cls,
                "alpha_mix": alpha,
                "p_fa": p_fa,
                "p_md": p_md,
                "p_ie": p_ie,
                "eval_s": dt,
            }
            rows.append(row)
            print(
                f"{name:<12} {w_rec:5.1f} {w_cls:5.1f} {alpha:5.1f} "
                f"{p_fa:10.2e} {p_md:10.2e} {p_ie:10.2e}  ({dt:.0f}s)"
            )

        a0, a1 = pies[0.0], pies[1.0]
        if np.isfinite(a0) and np.isfinite(a1):
            if a0 < a1 * 0.9:
                verdict = "alpha=0 mejor"
            elif a1 < a0 * 0.9:
                verdict = "alpha=1 mejor (!)"
            else:
                verdict = "~empate"
            print(f"  >> {name}: {verdict}  (P_IE alpha0={a0:.2e}, alpha1={a1:.2e})")

    print("=" * 72)
    print("Interpretación rápida: si alpha=0 gana en las 3 filas de pesos, sigue con alpha=0 en el TFG.")
    return rows


if __name__ == "__main__":
    run_mini_screening()
