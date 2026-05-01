import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _pca_2d(x):
    """
    Proyección PCA 2D sin dependencias externas.
    x: [N, D]
    """
    x = np.asarray(x, dtype=np.float32)
    x = x - np.mean(x, axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    return (u[:, :2] * s[:2]).astype(np.float32)


def _compute_global_block_error(u, u_hat, a_true, a_hat):
    """
    Error global por bloque (alineado con lógica de P_IE operacional):
      - MD en activo
      - FA en vacío
      - error de bits en activos detectados
    """
    a_true_b = a_true.squeeze() > 0.5
    a_hat_b = a_hat.squeeze() > 0.5

    blk_err = a_true_b & ~a_hat_b

    mask_ad = a_true_b & a_hat_b
    if np.any(mask_ad):
        u_ad = u[mask_ad]
        uhat_ad = np.clip(np.round(u_hat[mask_ad]), 0, 1)
        blk_ad_err = np.any(u_ad != uhat_ad, axis=-1)
        tmp = np.zeros_like(a_true_b, dtype=bool)
        tmp[np.where(mask_ad)[0]] = blk_ad_err
        blk_err = blk_err | tmp

    blk_err = blk_err | (~a_true_b & a_hat_b)
    return blk_err


def plot_latent_with_global_error(
    system,
    ebno_db,
    batch_size=3000,
    n_batches=2,
    use_pca=True,
    title=None,
    save_path=None,
):
    """
    Visualiza el espacio latente y rodea los puntos según acierto/fallo global.

    Requisitos:
      - `system` debe implementar `sample_with_latent(batch_size, ebno_db)`
        (incluido en ActivityAwarePolarSystem).

    Convención visual:
      - Color de relleno: clase real (activo/vacío)
      - Borde: rojo=falla global, negro=acierto global
    """
    if not hasattr(system, "sample_with_latent"):
        raise AttributeError(
            "El sistema no expone sample_with_latent(). "
            "Usa ActivityAwarePolarSystem actualizado."
        )

    zs, a_trues, a_hats, us, u_hats = [], [], [], [], []
    for _ in range(int(n_batches)):
        outputs = system.sample_with_latent(batch_size=int(batch_size), ebno_db=float(ebno_db))
        u, u_hat, _, _, a_true, _, a_hat, z = outputs
        zs.append(z)
        a_trues.append(a_true)
        a_hats.append(a_hat)
        us.append(u)
        u_hats.append(u_hat)

    z_all = np.concatenate(zs, axis=0)
    a_true_all = np.concatenate(a_trues, axis=0)
    a_hat_all = np.concatenate(a_hats, axis=0)
    u_all = np.concatenate(us, axis=0)
    u_hat_all = np.concatenate(u_hats, axis=0)

    err_global = _compute_global_block_error(u_all, u_hat_all, a_true_all, a_hat_all)
    a_true_b = a_true_all.squeeze() > 0.5

    if z_all.shape[1] < 2:
        raise ValueError(f"Latente con dimensión {z_all.shape[1]} < 2; no se puede representar 2D.")
    if use_pca:
        z2 = _pca_2d(z_all)
        xlab, ylab = "Latent PC1", "Latent PC2"
    else:
        z2 = z_all[:, :2]
        xlab, ylab = "Latent dim 1", "Latent dim 2"

    face_colors = np.where(a_true_b, "#1f77b4", "#ff7f0e")  # activo/vacio
    edge_colors = np.where(err_global, "red", "black")      # fallo/acierto global

    plt.figure(figsize=(8, 6))
    plt.scatter(
        z2[:, 0],
        z2[:, 1],
        s=18,
        c=face_colors,
        edgecolors=edge_colors,
        linewidths=0.5,
        alpha=0.8,
    )
    if title is None:
        title = f"Espacio latente | Eb/No={ebno_db:.2f} dB"
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.grid(alpha=0.25)

    legend_items = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4",
               markeredgecolor="black", markersize=7, label="A=1 (activo)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#ff7f0e",
               markeredgecolor="black", markersize=7, label="A=0 (vacío)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor="red", markersize=7, label="Fallo global"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor="black", markersize=7, label="Acierto global"),
    ]
    plt.legend(handles=legend_items, loc="best")

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figura guardada: {save_path}")

    plt.show()

    return {
        "n_points": int(len(a_true_b)),
        "p_error_global": float(np.mean(err_global)),
        "n_errors": int(np.sum(err_global)),
    }
