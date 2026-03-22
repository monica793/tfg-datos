import os
import tensorflow as tf
from sionna.phy.mapping import Constellation, Mapper, BinarySource
from sionna.phy.fec.polar import Polar5GEncoder
from sionna.phy.utils import ebnodb2no
from sionna.phy.channel import AWGN

from models.supervised_ae import SupervisedAE
from utils.signal import c2ri, rho_db_to_ebno_db


# ============================================================
# Hiperparámetros (modificar aquí o cargar desde config/hybrid.yaml)
# ============================================================
N_FIXED            = 100
P_EMPTY            = 0.30
RHO_DBS            = [0.0, 3.0]
K_CAND             = list(range(5, N_FIXED + 1, 5))

AE_EPOCHS          = 20
AE_STEPS_PER_EPOCH = 300
AE_BATCH_SIZE      = 256
LR                 = 2e-3
W_RECON            = 1.0
W_CLASS            = 10.0

CHECKPOINT_DIR     = "results/checkpoints/"


def k_is_valid_for_5g(k, n):
    """Comprueba si k es válido para Polar5GEncoder con longitud n."""
    try:
        Polar5GEncoder(k=int(k), n=int(n))
        return True
    except Exception:
        return False


def train_ae_for_n(n, rho_db, k_train):
    """
    Entrena el SupervisedAE para un bloque de longitud n.
    Si ya existe un checkpoint lo carga directamente sin reentrenar.
    Devuelve el modelo con trainable=False.
    """
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"ae_n{n}_rho{rho_db}.weights.h5")

    ae = SupervisedAE(n=n, latent_dim=64, hidden_dim=256, dropout=0.1)

    if os.path.exists(checkpoint_path):
        ae.load_weights(checkpoint_path)
        ae.trainable = False
        print(f"Checkpoint cargado: {checkpoint_path}")
        return ae

    # No hay checkpoint: entrenamos desde cero
    print(f"\n=== Entrenando AE | n={n} | rho={rho_db} dB | k_train={k_train} ===")

    rate_train    = k_train / n
    ebno_train_db = rho_db_to_ebno_db(rho_db, rate_train)

    src    = BinarySource()
    enc    = Polar5GEncoder(k=k_train, n=n)
    const  = Constellation("pam", num_bits_per_symbol=1, trainable=False)
    mapper = Mapper(constellation=const)
    ch     = AWGN()

    bce = tf.keras.losses.BinaryCrossentropy()
    mse = tf.keras.losses.MeanSquaredError()
    opt = tf.keras.optimizers.Adam(learning_rate=LR)

    @tf.function
    def make_batch(batch_size):
        a      = tf.cast(tf.random.uniform([batch_size, 1]) > P_EMPTY, tf.float32)
        u      = src([batch_size, k_train])
        c      = enc(u)
        x_info = mapper(c)
        x      = x_info * tf.cast(a, x_info.dtype)
        no     = ebnodb2no(tf.constant(ebno_train_db, tf.float32),
                           num_bits_per_symbol=1, coderate=rate_train)
        y      = ch(x, no)
        return c2ri(y), c2ri(x), a

    @tf.function
    def train_step(batch_size):
        y_ri, x_ri, a = make_batch(batch_size)
        with tf.GradientTape() as tape:
            x_hat_ri, p_active = ae(y_ri, training=True)
            loss = W_RECON * mse(x_ri, x_hat_ri) + W_CLASS * bce(a, p_active)
        grads = tape.gradient(loss, ae.trainable_variables)
        opt.apply_gradients(zip(grads, ae.trainable_variables))
        return loss

    for ep in range(1, AE_EPOCHS + 1):
        total_loss = sum(
            float(train_step(tf.constant(AE_BATCH_SIZE, tf.int32)).numpy())
            for _ in range(AE_STEPS_PER_EPOCH)
        )
        if ep % 5 == 0 or ep == 1 or ep == AE_EPOCHS:
            print(f"Epoch {ep:02d}/{AE_EPOCHS} | Loss={total_loss / AE_STEPS_PER_EPOCH:.4f}")

    # Guardar checkpoint para la próxima vez
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ae.save_weights(checkpoint_path)
    print(f"Checkpoint guardado: {checkpoint_path}")

    ae.trainable = False
    return ae


if __name__ == "__main__":
    for rho_db in RHO_DBS:
        valid_ks = [k for k in K_CAND if k < N_FIXED and k_is_valid_for_5g(k, N_FIXED)]
        k_train  = valid_ks[len(valid_ks) // 2]
        train_ae_for_n(n=N_FIXED, rho_db=rho_db, k_train=k_train)