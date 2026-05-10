import os
import numpy as np
import torch
import torch.nn as nn

from models.end_to_end import E2EEncoder, E2EDecoder
from utils.signal import rho_db_to_ebno_db


# ============================================================
# Hiperparámetros
# ============================================================
K               = 50       # bits de información
N               = 100      # símbolos transmitidos (rate = K/N = 0.5)
HIDDEN_DIM      = 256

P_EMPTY         = 0.30
RHO_DBS         = [0.0, 3.0]

EPOCHS          = 50
STEPS_PER_EPOCH = 300
BATCH_SIZE      = 256
LR              = 1e-3
W_BITS          = 10.0     # peso pérdida decodificación
W_ACT           = 1.0     # peso pérdida detección de actividad

CHECKPOINT_DIR  = "results/checkpoints/"


def train_e2e(k: int, n: int, rho_db: float):
    """
    Entrena el sistema end-to-end para una tasa R=k/n y un SNR rho_db.
    Si existe checkpoint lo carga directamente.
    Devuelve (encoder, decoder) listos para evaluar.
    """
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"e2e_k{k}_n{n}_rho{rho_db}.pt")

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = E2EEncoder(k=k, n=n, hidden_dim=HIDDEN_DIM).to(device)
    decoder = E2EDecoder(k=k, n=n, hidden_dim=HIDDEN_DIM).to(device)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        encoder.load_state_dict(checkpoint["encoder"])
        decoder.load_state_dict(checkpoint["decoder"])
        encoder.eval(); decoder.eval()
        print(f"Checkpoint cargado: {checkpoint_path}")
        return encoder, decoder

    print(f"\n=== Entrenando E2E | k={k} | n={n} | rho={rho_db} dB ===")

    ebno_db  = rho_db_to_ebno_db(rho_db, k / n)
    rate     = k / n
    no_lin   = 10 ** (-ebno_db / 10) / rate
    sigma    = np.sqrt(no_lin / 2)

    optimizer  = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=LR
    )
    bce = nn.BCELoss()

    encoder.train(); decoder.train()

    for ep in range(1, EPOCHS + 1):
        total_loss = 0.0

        for _ in range(STEPS_PER_EPOCH):
            # Generar batch
            a_np = (np.random.rand(BATCH_SIZE, 1) > P_EMPTY).astype(np.float32)
            u_np = np.random.randint(0, 2, (BATCH_SIZE, k)).astype(np.float32)

            a_pt = torch.tensor(a_np, device=device)
            u_pt = torch.tensor(u_np, device=device)

            # TX
            u_pm = u_pt * 2.0 - 1.0              # 0/1 → ±1
            x    = encoder(u_pm)                  # [B, n] ±1
            x    = x * a_pt                       # silencio si vacío

            # Canal AWGN (diferenciable — gradiente fluye hacia encoder)
            noise = torch.randn_like(x) * sigma
            y     = x + noise                     # [B, n]

            # RX
            u_hat, p_active = decoder(y)          # [B,k], [B,1]

            # Bits: solo ranuras con transmisión (A=1). En A=0 solo hay ruido; no hay mensaje que decodificar.
            mask = a_pt.squeeze(-1) > 0.5
            if mask.any():
                loss_bits = bce(u_hat[mask], u_pt[mask])
            else:
                loss_bits = torch.zeros((), device=device)

            loss_act  = bce(p_active, a_pt)
            loss      = W_BITS * loss_bits + W_ACT * loss_act

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if ep % 10 == 0 or ep == 1 or ep == EPOCHS:
            print(f"Epoch {ep:02d}/{EPOCHS} | Loss={total_loss / STEPS_PER_EPOCH:.4f}")

    # Guardar checkpoint
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save({
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "k": k, "n": n, "rho_db": rho_db
    }, checkpoint_path)
    print(f"Checkpoint guardado: {checkpoint_path}")

    encoder.eval(); decoder.eval()
    return encoder, decoder


def train_e2e_sweep_k(
    ks: list[int],
    n: int,
    rho_dbs: list[float] | None = None,
) -> dict[float, dict[int, tuple["E2EEncoder", "E2EDecoder"]]]:
    """
    Entrena (o carga) un par encoder/decoder por cada (k, rho_db).
    Orden: para cada k se completan todos los rho_db antes de pasar al siguiente k.

    Checkpoints: results/checkpoints/e2e_k{k}_n{n}_rho{rho_db}.pt

    Devuelve: e2e_models[rho_db][k] = (encoder, decoder)
    """
    if rho_dbs is None:
        rho_dbs = list(RHO_DBS)
    out: dict[float, dict[int, tuple[E2EEncoder, E2EDecoder]]] = {}
    for rho_db in rho_dbs:
        out[float(rho_db)] = {}
    for k in ks:
        for rho_db in rho_dbs:
            rho_f = float(rho_db)
            enc, dec = train_e2e(k=int(k), n=int(n), rho_db=rho_f)
            out[rho_f][int(k)] = (enc, dec)
    return out


if __name__ == "__main__":
    for rho_db in RHO_DBS:
        train_e2e(k=K, n=N, rho_db=rho_db)