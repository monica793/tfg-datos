import os
import tensorflow as tf
from sionna.phy.mapping import Constellation, Mapper, BinarySource
from sionna.phy.fec.polar import Polar5GEncoder
from sionna.phy.utils import ebnodb2no
from sionna.phy.channel import AWGN
import numpy as np
import random
try:
    import wandb
except Exception:
    wandb = None

from models.supervised_ae import SupervisedAE
from utils.signal import c2ri, rho_db_to_ebno_db


# ============================================================
# Hiperparámetros
# ============================================================
N_FIXED            = 100
P_EMPTY            = 0.30
RHO_DBS            = [0.0, 3.0]
K_CAND             = list(range(5, N_FIXED + 1, 5))

AE_EPOCHS          = 20
AE_STEPS_PER_EPOCH = 300
AE_BATCH_SIZE      = 256   # entero Python, no tf.constant
LR                 = 2e-3
W_RECON            = 1.0
W_CLASS            = 10.0

CHECKPOINT_DIR     = "results/checkpoints/"
SEED               = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def k_is_valid_for_5g(k, n):
    try:
        Polar5GEncoder(k=int(k), n=int(n))
        return True
    except Exception:
        return False


def make_batch(src, enc, mapper, ebno_train_db, rate_train, batch_size):
    """Genera un batch de datos. batch_size es un entero Python normal."""
    a = (np.random.rand(batch_size, 1) > P_EMPTY).astype("float32")
    a_tf = tf.constant(a)

    u      = src([batch_size, enc.k])
    c      = enc(u)
    x_info = mapper(c)

    # Sionna devuelve tensores PyTorch — convertimos a numpy y luego a TF
    import torch
    x_info_np = x_info.detach().cpu().numpy()
    x_info_tf = tf.complex(
        tf.constant(x_info_np.real, dtype=tf.float32),
        tf.constant(x_info_np.imag, dtype=tf.float32)
    )

    x_tf = x_info_tf * tf.cast(a_tf, x_info_tf.dtype)

    # Canal AWGN de Sionna sobre numpy
    x_np   = x_tf.numpy()
    no_val = 10 ** (-ebno_train_db / 10) / rate_train
    noise  = np.sqrt(no_val / 2) * (np.random.randn(*x_np.shape) + 1j * np.random.randn(*x_np.shape))
    y_np   = x_np + noise.astype(x_np.dtype)

    y_tf = tf.complex(
        tf.constant(y_np.real, dtype=tf.float32),
        tf.constant(y_np.imag, dtype=tf.float32)
    )

    return c2ri(y_tf), c2ri(x_tf), a_tf


def train_ae_for_n(
    n,
    rho_db,
    k_train=None,
    valid_ks=None,
    use_wandb=True,
    wandb_project="tfg-datos-hybrid-ae",
    checkpoint_tag=None,
    latent_dim=64,
    hidden_dim=256,
    dropout=0.1,
    ae_epochs=None,
    ae_steps_per_epoch=None,
    ae_batch_size=None,
    lr=None,
    w_recon=None,
    w_class=None,
):
    """
    Entrena el SupervisedAE para un bloque de longitud n.
    - Si valid_ks tiene varios valores, entrena en modo multi-k (generaliza en tasa).
    - Si no, usa k_train o el k central válido como fallback.
    Si ya existe un checkpoint lo carga directamente sin reentrenar.
    """
    if valid_ks is None:
        valid_ks = [k for k in K_CAND if k < n and k_is_valid_for_5g(k, n)]
    valid_ks = sorted(set(int(k) for k in valid_ks if k < n and k_is_valid_for_5g(k, n)))
    if not valid_ks:
        raise ValueError(f"No hay k válidos para n={n}")

    if k_train is not None:
        # Compatibilidad hacia atrás: si se pasa explícitamente k_train, se fuerza single-k
        valid_ks = [int(k_train)]

    mode_tag = "multik" if len(valid_ks) > 1 else f"k{valid_ks[0]}"
    if checkpoint_tag is not None and str(checkpoint_tag).strip():
        mode_tag = f"{mode_tag}_{str(checkpoint_tag).strip().replace(' ', '_')}"
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"ae_n{n}_rho{rho_db}_{mode_tag}.weights.h5")

    ae_epochs = int(AE_EPOCHS if ae_epochs is None else ae_epochs)
    ae_steps_per_epoch = int(AE_STEPS_PER_EPOCH if ae_steps_per_epoch is None else ae_steps_per_epoch)
    ae_batch_size = int(AE_BATCH_SIZE if ae_batch_size is None else ae_batch_size)
    lr = float(LR if lr is None else lr)
    w_recon = float(W_RECON if w_recon is None else w_recon)
    w_class = float(W_CLASS if w_class is None else w_class)

    ae = SupervisedAE(n=n, latent_dim=int(latent_dim), hidden_dim=int(hidden_dim), dropout=float(dropout))

    if os.path.exists(checkpoint_path):
        # Keras requiere un forward pass antes de cargar pesos
        dummy = tf.zeros([1, n, 2])
        ae(dummy, training=False)
        ae.load_weights(checkpoint_path)
        ae.trainable = False
        print(f"Checkpoint cargado: {checkpoint_path}")
        return ae

    print(f"\n=== Entrenando AE | n={n} | rho={rho_db} dB | ks={valid_ks} ===")

    wandb_run = None
    if use_wandb and wandb is not None:
        try:
            wandb_run = wandb.init(
                project=wandb_project,
                reinit=True,
                config={
                    "n": n,
                    "rho_db": rho_db,
                    "valid_ks": valid_ks,
                    "epochs": ae_epochs,
                    "steps_per_epoch": ae_steps_per_epoch,
                    "batch_size": ae_batch_size,
                    "lr": lr,
                    "w_recon": w_recon,
                    "w_class": w_class,
                    "latent_dim": int(latent_dim),
                    "hidden_dim": int(hidden_dim),
                    "dropout": float(dropout),
                    "seed": SEED,
                    "mode_tag": mode_tag,
                },
                name=f"ae_n{n}_rho{rho_db}_{mode_tag}",
            )
        except Exception as e:
            print(f"[WARN] No se pudo iniciar wandb ({type(e).__name__}: {e}). Continúa sin logging.")
            wandb_run = None

    src = BinarySource()
    const = Constellation("pam", num_bits_per_symbol=1, trainable=False)
    encoders = {k: Polar5GEncoder(k=k, n=n) for k in valid_ks}
    mappers  = {k: Mapper(constellation=const) for k in valid_ks}

    bce = tf.keras.losses.BinaryCrossentropy()
    mse = tf.keras.losses.MeanSquaredError()
    opt = tf.keras.optimizers.Adam(learning_rate=lr)

    for ep in range(1, ae_epochs + 1):
        total_loss = 0.0
        total_recon = 0.0
        total_class = 0.0
        for _ in range(ae_steps_per_epoch):
            k_step = random.choice(valid_ks)
            rate_step = k_step / n
            ebno_step = rho_db_to_ebno_db(rho_db, rate_step)
            y_ri, x_ri, a = make_batch(src, encoders[k_step], mappers[k_step],
                                       ebno_step, rate_step, ae_batch_size)
            with tf.GradientTape() as tape:
                x_hat_ri, p_active = ae(y_ri, training=True)
                loss_recon = mse(x_ri, x_hat_ri)
                loss_class = bce(a, p_active)
                loss = w_recon * loss_recon + w_class * loss_class
            grads = tape.gradient(loss, ae.trainable_variables)
            opt.apply_gradients(zip(grads, ae.trainable_variables))
            total_loss += float(loss.numpy())
            total_recon += float(loss_recon.numpy())
            total_class += float(loss_class.numpy())

        avg_loss = total_loss / ae_steps_per_epoch
        avg_recon = total_recon / ae_steps_per_epoch
        avg_class = total_class / ae_steps_per_epoch
        if ep % 5 == 0 or ep == 1 or ep == ae_epochs:
            print(
                f"Epoch {ep:02d}/{ae_epochs} | "
                f"Loss={avg_loss:.4f} | Recon={avg_recon:.4f} | Class={avg_class:.4f}"
            )

        if wandb_run is not None:
            wandb.log({
                "train/epoch": ep,
                "train/loss_total": avg_loss,
                "train/loss_recon": avg_recon,
                "train/loss_class": avg_class,
                "train/rho_db": float(rho_db),
                "train/n": int(n),
                "train/n_valid_ks": int(len(valid_ks)),
            })

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ae.save_weights(checkpoint_path)
    print(f"Checkpoint guardado: {checkpoint_path}")

    if use_wandb and wandb is None:
        print("[WARN] wandb no está instalado. Continúa sin logging.")

    if wandb_run is not None:
        wandb.log({"train/checkpoint_saved": 1, "train/checkpoint_path": checkpoint_path})
        wandb.finish()

    ae.trainable = False
    return ae


if __name__ == "__main__":
    for rho_db in RHO_DBS:
        valid_ks = [k for k in K_CAND if k < N_FIXED and k_is_valid_for_5g(k, N_FIXED)]
        train_ae_for_n(n=N_FIXED, rho_db=rho_db, valid_ks=valid_ks)