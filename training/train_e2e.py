import csv
import os

import numpy as np
import torch
import torch.nn as nn

try:
    import wandb as _wandb
except ImportError:
    _wandb = None

from models.end_to_end import E2EEncoder, E2EDecoder
from utils.signal import rho_db_to_ebno_db


# ============================================================
# Hiperparámetros
# ============================================================
K               = 50
N               = 100
HIDDEN_DIM      = 256

P_EMPTY         = 0.30
RHO_DBS         = [0.0, 3.0]

EPOCHS          = 120      # 50 era insuficiente para k grande
STEPS_PER_EPOCH = 300
BATCH_SIZE      = 256
LR              = 1e-3
W_BITS          = 10.0
W_ACT           = 1.0

GRAD_CLIP       = 1.0      # max_norm para gradient clipping (STE)

# ReduceLROnPlateau sobre loss_bits
LR_PATIENCE     = 8        # épocas sin mejora en loss_bits antes de reducir LR
LR_FACTOR       = 0.5      # factor de reducción
LR_MIN          = 1e-5     # LR mínimo

CHECKPOINT_DIR   = "results/checkpoints/"
TRAINING_LOG_DIR = "results/training_logs"


def _save_history_csv(history: list[dict], path: str) -> None:
    if not history:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def train_e2e(
    k: int,
    n: int,
    rho_db: float,
    force_retrain: bool = False,
    use_wandb: bool = False,
    wandb_project: str = "tfg-datos-e2e",
) -> tuple[E2EEncoder, E2EDecoder]:
    """
    Entrena el sistema end-to-end para R=k/n y SNR rho_db.
    Mejoras respecto a la versión original:
      - ReduceLROnPlateau sobre loss_bits (no sobre la total)
      - Gradient clipping (estabiliza STE)
      - Log separado de loss_bits / loss_act
      - CSV de historial en results/training_logs/
      - force_retrain para ignorar checkpoint existente
      - W&B opcional
    Devuelve (encoder, decoder) listos para evaluar.
    """
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"e2e_k{k}_n{n}_rho{rho_db}.pt")
    run_stem = f"e2e_k{k}_n{n}_rho{rho_db}"
    history_path = os.path.join(TRAINING_LOG_DIR, f"{run_stem}_history.csv")

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = E2EEncoder(k=k, n=n, hidden_dim=HIDDEN_DIM).to(device)
    decoder = E2EDecoder(k=k, n=n, hidden_dim=HIDDEN_DIM).to(device)

    if os.path.exists(checkpoint_path) and not force_retrain:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        encoder.load_state_dict(checkpoint["encoder"])
        decoder.load_state_dict(checkpoint["decoder"])
        encoder.eval(); decoder.eval()
        print(f"Checkpoint cargado: {checkpoint_path}")
        if os.path.exists(history_path):
            print(f"Historial: {history_path}")
        return encoder, decoder

    if force_retrain and os.path.exists(checkpoint_path):
        print(f"[force_retrain] Ignorando checkpoint existente: {checkpoint_path}")

    print(f"\n=== Entrenando E2E | k={k} | n={n} | R={k/n:.2f} | rho={rho_db} dB ===")

    ebno_db = rho_db_to_ebno_db(rho_db, k / n)
    sigma   = float(np.sqrt(10 ** (-ebno_db / 10) / (k / n) / 2))

    params    = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        min_lr=LR_MIN,
    )
    bce = nn.BCELoss()

    wandb_run = None
    if use_wandb and _wandb is not None:
        try:
            wandb_run = _wandb.init(
                project=wandb_project,
                reinit=True,
                name=run_stem,
                config={
                    "k": k, "n": n, "R": k / n, "rho_db": rho_db,
                    "epochs": EPOCHS, "steps_per_epoch": STEPS_PER_EPOCH,
                    "batch_size": BATCH_SIZE, "lr": LR, "w_bits": W_BITS,
                    "w_act": W_ACT, "grad_clip": GRAD_CLIP,
                    "lr_patience": LR_PATIENCE, "lr_factor": LR_FACTOR,
                    "hidden_dim": HIDDEN_DIM,
                },
                tags=["e2e", f"k{k}", f"rho{rho_db}"],
            )
        except Exception as e:
            print(f"[WARN] W&B no disponible: {e}")
            wandb_run = None

    encoder.train(); decoder.train()
    history: list[dict] = []

    for ep in range(1, EPOCHS + 1):
        total_loss = total_bits = total_act = 0.0
        n_steps_with_bits = 0

        for _ in range(STEPS_PER_EPOCH):
            a_np = (np.random.rand(BATCH_SIZE, 1) > P_EMPTY).astype(np.float32)
            u_np = np.random.randint(0, 2, (BATCH_SIZE, k)).astype(np.float32)

            a_pt = torch.tensor(a_np, device=device)
            u_pt = torch.tensor(u_np, device=device)

            u_pm = u_pt * 2.0 - 1.0
            x    = encoder(u_pm) * a_pt
            y    = x + torch.randn_like(x) * sigma

            u_hat, p_active = decoder(y)

            mask = a_pt.squeeze(-1) > 0.5
            if mask.any():
                loss_bits = bce(u_hat[mask], u_pt[mask])
                n_steps_with_bits += 1
            else:
                loss_bits = torch.zeros((), device=device)

            loss_act  = bce(p_active, a_pt)
            loss      = W_BITS * loss_bits + W_ACT * loss_act

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=GRAD_CLIP)
            optimizer.step()

            total_loss += loss.item()
            total_bits += loss_bits.item()
            total_act  += loss_act.item()

        denom_bits = max(n_steps_with_bits, 1)
        avg_total = total_loss / STEPS_PER_EPOCH
        avg_bits  = total_bits / denom_bits
        avg_act   = total_act  / STEPS_PER_EPOCH
        current_lr = optimizer.param_groups[0]["lr"]

        # Scheduler decide si reducir LR basándose en loss_bits
        scheduler.step(avg_bits)

        row = {
            "epoch": ep,
            "loss_total": avg_total,
            "loss_bits": avg_bits,
            "loss_act": avg_act,
            "lr": current_lr,
        }
        history.append(row)

        # Imprimir cada 5 épocas (y la primera y última)
        if ep % 5 == 0 or ep == 1 or ep == EPOCHS:
            print(
                f"Epoch {ep:03d}/{EPOCHS} | "
                f"loss_total={avg_total:.4f} | "
                f"loss_bits={avg_bits:.4f} | "
                f"loss_act={avg_act:.4f} | "
                f"lr={current_lr:.2e}"
            )

        if wandb_run is not None:
            _wandb.log({
                "epoch": ep,
                "train/loss_total": avg_total,
                "train/loss_bits": avg_bits,
                "train/loss_act": avg_act,
                "train/lr": current_lr,
            })

    _save_history_csv(history, history_path)
    print(f"Historial guardado: {history_path}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(
        {"encoder": encoder.state_dict(), "decoder": decoder.state_dict(),
         "k": k, "n": n, "rho_db": rho_db},
        checkpoint_path,
    )
    print(f"Checkpoint guardado: {checkpoint_path}")

    if wandb_run is not None:
        _wandb.finish()

    encoder.eval(); decoder.eval()
    return encoder, decoder


def train_e2e_sweep_k(
    ks: list[int],
    n: int,
    rho_dbs: list[float] | None = None,
    force_retrain: bool = False,
    use_wandb: bool = False,
) -> dict[float, dict[int, tuple[E2EEncoder, E2EDecoder]]]:
    """
    Entrena (o carga) un par encoder/decoder por cada (k, rho_db).
    Checkpoints: results/checkpoints/e2e_k{k}_n{n}_rho{rho_db}.pt
    Devuelve: e2e_models[rho_db][k] = (encoder, decoder)
    """
    if rho_dbs is None:
        rho_dbs = list(RHO_DBS)
    out: dict[float, dict[int, tuple[E2EEncoder, E2EDecoder]]] = {
        float(rho): {} for rho in rho_dbs
    }
    for k in ks:
        for rho_db in rho_dbs:
            enc, dec = train_e2e(
                k=int(k), n=int(n), rho_db=float(rho_db),
                force_retrain=force_retrain,
                use_wandb=use_wandb,
            )
            out[float(rho_db)][int(k)] = (enc, dec)
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Entrena E2E MLP.")
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--rho-db", type=float, default=None)
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    rho_list = [args.rho_db] if args.rho_db is not None else RHO_DBS
    for rho_db in rho_list:
        train_e2e(
            k=args.k, n=N, rho_db=rho_db,
            force_retrain=args.force_retrain,
            use_wandb=args.wandb,
        )
