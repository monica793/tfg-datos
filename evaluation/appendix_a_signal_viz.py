"""
Apéndice A — Visualización del aplastamiento de ruido por el Autoencoder
=========================================================================
Genera una figura con 3 subplots apilados verticalmente que muestra
los primeros N_SYMBOLS símbolos de un paquete activo:

  (a) Señal BPSK transmitida x  — valores ±1
  (b) Señal ruidosa recibida y  — con ruido AWGN
  (c) Señal reconstruida x̂     — salida del AE (denoising)

La figura ilustra cómo el AE comprime la varianza del ruido, perdiendo
la información suave de los LLRs necesaria para el decodificador Polar.

Uso en Colab:
    from evaluation.appendix_a_signal_viz import plot_signal_chain
    from training.train_hybrid import train_ae_for_n, N_FIXED, K_CAND, k_is_valid_for_5g

    valid_ks = [k for k in K_CAND if k < N_FIXED and k_is_valid_for_5g(k, N_FIXED)]
    ae = train_ae_for_n(n=N_FIXED, rho_db=3.0, valid_ks=valid_ks)
    plot_signal_chain(ae, rho_db=0.0)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from evaluation.plot_pfa_pmd_pie import apply_thesis_style
from systems.hybrid_polar import ActivityAwarePolarSystem
from training.train_hybrid import N_FIXED, K_CAND, k_is_valid_for_5g
from utils.signal import rho_db_to_ebno_db


# Número de símbolos a mostrar en la figura
N_SYMBOLS = 50
Y_LIM = (-2.5, 2.5)


def _stem_discrete(ax, idx, values, color, marker="o", markersize=4.0, linewidth=0.9):
    """Stem plot for discrete symbol sequences (communications-style)."""
    markerline, stemlines, baseline = ax.stem(idx, values, basefmt=" ")
    plt.setp(stemlines, color=color, linewidth=linewidth, alpha=0.85, zorder=2)
    plt.setp(
        markerline,
        color=color,
        marker=marker,
        markersize=markersize,
        markeredgewidth=0.9,
        zorder=3,
    )
    baseline.set_visible(False)


def plot_signal_chain(
    ae,
    rho_db: float = 0.0,
    n_symbols: int = N_SYMBOLS,
    batch_size: int = 300,
    k: int = None,
    seed: int = 42,
    figure_path: str = None,
    show: bool = True,
):
    """
    Genera la figura de 3 subplots para el Apéndice A.

    Parámetros
    ----------
    ae          : modelo SupervisedAE ya entrenado
    rho_db      : SNR por símbolo en dB (se recomienda 0.0 para ilustrar ruido intenso)
    n_symbols   : número de símbolos a mostrar en el eje X (por defecto 50)
    batch_size  : tamaño del batch para buscar un paquete activo
    k           : dimensión del código. Si None, se elige la k central de los válidos.
    seed        : semilla para reproducibilidad
    figure_path : ruta de guardado. Si None, se usa results/figures/
    show        : si True, llama a plt.show()
    """
    np.random.seed(seed)
    apply_thesis_style()

    # Seleccionar k
    valid_ks = [kk for kk in K_CAND if kk < N_FIXED and k_is_valid_for_5g(kk, N_FIXED)]
    if not valid_ks:
        raise RuntimeError("No hay k válidos para Polar5G con n=%d" % N_FIXED)
    if k is None:
        k = valid_ks[len(valid_ks) // 2]

    ebno_db = rho_db_to_ebno_db(rho_db, k / N_FIXED)
    n_sym   = min(n_symbols, N_FIXED)
    sym_idx = np.arange(n_sym)

    system = ActivityAwarePolarSystem(
        k=k, n=N_FIXED, ae_model=ae, p_empty=0.3, thresh=0.5, alpha_mix=1.0
    )

    # Obtener señales intermedias; buscar hasta 5 batches para un paquete activo
    x_plot = y_plot = xhat_plot = None
    for attempt in range(5):
        x_clean, y_noisy, x_hat, a = system.sample_signal_chain(batch_size, ebno_db)
        active_idx = np.where(a.squeeze() > 0.5)[0]
        if len(active_idx) > 0:
            idx = active_idx[0]
            # squeeze por si hay dimensión extra (ej. shape [n,1] -> [n])
            x_plot    = np.real(x_clean[idx]).squeeze()[:n_sym]
            y_plot    = np.real(y_noisy[idx]).squeeze()[:n_sym]
            xhat_plot = np.real(x_hat[idx]).squeeze()[:n_sym]
            break

    if x_plot is None:
        raise RuntimeError(
            "No se encontró ningún paquete activo en %d intentos con batch_size=%d. "
            "Prueba a reducir p_empty o aumentar batch_size." % (5, batch_size)
        )

    # Métricas de varianza
    var_x    = float(np.var(x_plot))
    var_y    = float(np.var(y_plot))
    var_xhat = float(np.var(xhat_plot))
    ratio    = var_y / var_xhat if var_xhat > 0 else float("inf")

    # ------------------------------------------------------------------ figura
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
    fig.suptitle(
        f"Noise variance compression by the Autoencoder  "
        f"($\\rho = {rho_db:.0f}$ dB,  $n = {N_FIXED}$,  $k = {k}$,  "
        f"$R = {k/N_FIXED:.2f}$)",
        fontsize=13,
    )

    COLORS = ["#2166ac", "#d73027", "#1a9641"]   # azul, rojo, verde

    # Líneas de referencia BPSK (±1)
    for ax in axes:
        ax.axhline( 1.0, color="gray", linewidth=0.7, linestyle=":", alpha=0.6,
                    zorder=1)
        ax.axhline(-1.0, color="gray", linewidth=0.7, linestyle=":", alpha=0.6,
                    zorder=1)
        ax.axhline( 0.0, color="gray", linewidth=0.4, linestyle="-", alpha=0.3,
                    zorder=1)

    # Subplot (a) — señal transmitida
    _stem_discrete(axes[0], sym_idx, x_plot, COLORS[0], marker="o", markersize=4.0)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(
        f"(a) Transmitted signal $x$ — BPSK  "
        f"[$\\sigma_x^2 = {var_x:.4f}$]",
        loc="left",
    )
    axes[0].set_ylim(Y_LIM)

    # Subplot (b) — señal ruidosa
    _stem_discrete(axes[1], sym_idx, y_plot, COLORS[1], marker="o", markersize=3.5, linewidth=0.8)
    axes[1].set_ylabel("Amplitude")
    axes[1].set_title(
        f"(b) Noisy received signal $y$  "
        f"[$\\sigma_y^2 = {var_y:.4f}$]",
        loc="left",
    )
    axes[1].set_ylim(Y_LIM)

    # Subplot (c) — señal reconstruida por AE
    _stem_discrete(axes[2], sym_idx, xhat_plot, COLORS[2], marker="x", markersize=4.5)
    axes[2].set_ylabel("Amplitude")
    axes[2].set_xlabel("Symbol index")
    axes[2].set_title(
        f"(c) AE reconstructed signal $\\hat{{x}}$  "
        f"[$\\sigma_{{\\hat{{x}}}}^2 = {var_xhat:.4f}$;  "
        f"$\\sigma_y^2 = {ratio:.1f}\\,\\sigma_{{\\hat{{x}}}}^2$]",
        loc="left",
    )
    axes[2].set_ylim(Y_LIM)

    for ax in axes:
        ax.set_xlim([0, n_sym - 1])
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))

    fig.tight_layout()

    # Guardar
    if figure_path is None:
        os.makedirs("results/figures", exist_ok=True)
        figure_path = (
            f"results/figures/appendix_a_signal_chain_rho{rho_db:.0f}_k{k}.png"
        )
    else:
        os.makedirs(os.path.dirname(figure_path) or ".", exist_ok=True)

    fig.savefig(figure_path)
    print(f"Figura guardada: {figure_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {"var_x": var_x, "var_y": var_y, "var_xhat": var_xhat,
            "compression_ratio": ratio}


# ---------------------------------------------------------------------------
# Bloque __main__: permite ejecutar directamente como script o desde Colab
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from training.train_hybrid import train_ae_for_n, RHO_DBS

    valid_ks = [k for k in K_CAND if k < N_FIXED and k_is_valid_for_5g(k, N_FIXED)]

    # Entrenar (o cargar checkpoint) con rho=3 dB como en los resultados principales
    ae = train_ae_for_n(n=N_FIXED, rho_db=3.0, valid_ks=valid_ks)

    # Figura de aplastamiento a rho=0 dB (SNR bajo, ruido más visible)
    stats = plot_signal_chain(ae, rho_db=0.0)
    print(
        f"\nResumen varianzas:"
        f"\n  σ²(x)    = {stats['var_x']:.4f}"
        f"\n  σ²(y)    = {stats['var_y']:.4f}"
        f"\n  σ²(x̂)   = {stats['var_xhat']:.4f}"
        f"\n  ratio    = {stats['compression_ratio']:.1f}×"
    )
