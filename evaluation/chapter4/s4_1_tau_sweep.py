"""
Barrido del umbral de decision tau — Trade-off P_FA / P_MD.

Subseccion 4.1 de la memoria (previa al barrido de tasa R=k/n).

Configuracion fija
------------------
n = 100, k = 50  ->  R = 0.5  (punto central del barrido principal)
p_empty  = 0.30
alpha_mix = 0  (sin denoising del AE; justificado en Apendice A)
SNR: rho in {0.0, 3.0} dB

Estrategia de evaluacion eficiente
-----------------------------------
La estadistica de actividad (p_active, a_true) se recopila en UN UNICO
pase por la red para cada par (sistema, rho).  A partir de ese pool se
barre tau analiticamente, evitando re-simular el canal para cada valor.
Esto hace la evaluacion N_THRESHOLDS veces mas rapida que el metodo naive
y garantiza que todas las curvas ven exactamente las mismas realizaciones
del canal, haciendo la comparacion estadisticamente mas justa.
"""
from __future__ import annotations

import csv
import os
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

# ---------------------------------------------------------------------------
# Configuracion por defecto
# ---------------------------------------------------------------------------
N_DEFAULT         = 100
K_DEFAULT         = 50
P_EMPTY_DEFAULT   = 0.30
RHO_DBS_DEFAULT   = (0.0, 3.0)
THRESHOLDS_DEFAULT = np.round(np.linspace(0.05, 0.95, 19), 4)

N_BATCHES_DEFAULT  = 100
BATCH_SIZE_DEFAULT = 5000

FIG_DIR   = 'results/figures'
TABLE_DIR = 'results/tables'

# ---------------------------------------------------------------------------
# Estilo de figura (memoria / publicacion)
# ---------------------------------------------------------------------------
_THESIS_RC = {
    'figure.facecolor':   'white',
    'axes.facecolor':     'white',
    'axes.edgecolor':     '#333333',
    'axes.grid':          True,
    'grid.color':         '#bbbbbb',
    'grid.linestyle':     '--',
    'grid.alpha':         0.3,
    'grid.linewidth':     0.6,
    'axes.axisbelow':     True,
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'Times', 'Nimbus Roman', 'DejaVu Serif'],
    'mathtext.fontset':   'stix',
    'font.size':          11,
    'axes.titlesize':     12,
    'axes.labelsize':     11,
    'xtick.labelsize':    10,
    'ytick.labelsize':    10,
    'legend.fontsize':    9.5,
    'legend.framealpha':  0.85,
    'lines.linewidth':    1.6,
    'lines.markersize':   5,
    'figure.dpi':         150,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.facecolor':  'white',
}

# ---------------------------------------------------------------------------
# Etiquetas LaTeX
# Usamos chr(92) = backslash para evitar literales con backslash en el
# codigo fuente (facilita el manejo en notebooks Colab y en JSON).
# ---------------------------------------------------------------------------
_B = chr(92)

L_TAU_XLABEL  = 'Decision threshold $' + _B + 'tau$'
L_PROB_YLABEL = 'Probability'
L_RHO = {
    0.0: '$' + _B + 'rho = 0' + _B + ';' + _B + 'mathrm{dB}$',
    3.0: '$' + _B + 'rho = 3' + _B + ';' + _B + 'mathrm{dB}$',
}
L_TAU05 = '$' + _B + 'tau = 0.5$'
L_SUPTITLE = (
    'Detection threshold trade-off  '
    '$n=100,' + _B + '; k=50,' + _B + '; '
    'p_{' + _B + 'mathrm{empty}} = 0.3$'
)

# Colores: consistentes con el resto de la memoria
C_HYBRID = '#1f77b4'   # azul C0 de matplotlib
C_E2E    = '#d62728'   # rojo C3 de matplotlib


# ---------------------------------------------------------------------------
# Helpers de simulacion
# ---------------------------------------------------------------------------

def collect_p_active(
    system,
    ebno_db: float,
    n_batches: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Ejecuta el sistema durante n_batches lotes y devuelve:
      - p_active : array [N] con la probabilidad continua de actividad (pre-umbral)
      - a_true   : array bool [N] con la actividad real

    Compatible con ambas arquitecturas:
      5 salidas  -> E2E   (u, u_hat, a_true, p_active, a_hat)
      7 salidas  -> Hibrido (u, u_hat, c, c_hat, a_true, p_active, a_hat)
    """
    all_p, all_a = [], []
    for _ in range(n_batches):
        outputs = system(batch_size=int(batch_size), ebno_db=float(ebno_db))
        if len(outputs) == 5:
            _, _, a_true, p_active, _ = outputs
        else:
            _, _, _, _, a_true, p_active, _ = outputs
        all_p.append(np.asarray(p_active).ravel())
        all_a.append(np.asarray(a_true).ravel() > 0.5)
    return np.concatenate(all_p), np.concatenate(all_a).astype(bool)


def sweep_thresholds(
    p_active: np.ndarray,
    a_true: np.ndarray,
    thresholds: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula P_FA y P_MD para cada valor de tau sobre el pool pre-recopilado.

    Sustituye P=0.0 por np.nan: en escala logaritmica un cero exacto colapsa
    a -inf, lo que rompe la linea. Con nan, matplotlib omite el punto y
    mantiene la linea continua en los tramos con datos validos.
    """
    n_empty  = int(np.sum(~a_true))
    n_active = int(np.sum( a_true))
    pfas: list[float] = []
    pmds: list[float] = []
    for tau in thresholds:
        a_hat = p_active > float(tau)
        fa = float(np.sum(~a_true & a_hat)) / n_empty  if n_empty  > 0 else np.nan
        md = float(np.sum( a_true & ~a_hat)) / n_active if n_active > 0 else np.nan
        pfas.append(np.nan if (not np.isfinite(fa) or fa == 0.0) else fa)
        pmds.append(np.nan if (not np.isfinite(md) or md == 0.0) else md)
    return np.array(pfas, dtype=float), np.array(pmds, dtype=float)


# ---------------------------------------------------------------------------
# Funcion principal
# ---------------------------------------------------------------------------

def plot_tau_sweep(
    n: int = N_DEFAULT,
    k: int = K_DEFAULT,
    p_empty: float = P_EMPTY_DEFAULT,
    rho_dbs: tuple = RHO_DBS_DEFAULT,
    thresholds: np.ndarray | None = None,
    n_batches: int = N_BATCHES_DEFAULT,
    batch_size: int = BATCH_SIZE_DEFAULT,
    fig_dir: str = FIG_DIR,
    show: bool = True,
    alpha_mix: float = 0.0,
) -> dict:
    """
    Carga/entrena los modelos Hibrido y E2E, ejecuta el barrido de tau
    y genera la figura de publicacion junto con un CSV de resultados.

    Parametros
    ----------
    n, k        : dimensiones del bloque (fijas para este analisis)
    p_empty     : fraccion de paquetes vacios
    rho_dbs     : SNRs de operacion (dB por simbolo)
    thresholds  : vector de tau; por defecto linspace(0.05, 0.95, 19)
    n_batches   : numero de lotes de simulacion por (sistema, rho)
    batch_size  : tamano de cada lote
    fig_dir     : directorio donde se guardan las figuras
    show        : si True llama plt.show(); False cierra la figura (Colab batch)
    alpha_mix   : mezcla AE/recibida para el hibrido (0=sin denoising, 1=AE puro)

    Devuelve
    --------
    results : dict  {(rho_db, 'Hybrid'|'E2E'): (pfas_array, pmds_array)}
    """
    from training.train_hybrid import train_ae_for_n, k_is_valid_for_5g
    from training.train_e2e   import train_e2e
    from systems.hybrid_polar import ActivityAwarePolarSystem
    from systems.e2e_system   import E2ESystem
    from utils.signal         import rho_db_to_ebno_db

    if thresholds is None:
        thresholds = THRESHOLDS_DEFAULT

    plt.rcParams.update(_THESIS_RC)
    os.makedirs(fig_dir,   exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)

    # ── Cargar / entrenar modelos ────────────────────────────────────────────
    print('=' * 65)
    print('Cargando modelos Hibrido (SupervisedAE multi-k) ...')
    valid_ks_h = [
        kk for kk in range(5, n + 1, 5)
        if kk < n and k_is_valid_for_5g(kk, n)
    ]
    hybrid_aes = {
        rho: train_ae_for_n(n=n, rho_db=rho, valid_ks=valid_ks_h)
        for rho in rho_dbs
    }

    print('Cargando modelos E2E ...')
    e2e_models = {
        rho: train_e2e(k=k, n=n, rho_db=rho)
        for rho in rho_dbs
    }
    print('Modelos listos.')
    print('=' * 65)

    # ── Barrido por SNR ──────────────────────────────────────────────────────
    results: dict = {}

    for rho_db in rho_dbs:
        ebno_db = rho_db_to_ebno_db(rho_db, k / n)
        total   = n_batches * batch_size
        print(f'\nrho = {rho_db} dB  ->  Eb/No = {ebno_db:.2f} dB'
              f'  |  k={k}, n={n}, R={k/n:.2f}')

        # Hibrido
        # alpha_mix=0 -> la senal que entra al decoder Polar es la recibida
        # sin denoising. El AE sigue corriendo para obtener p_active;
        # la rama de deteccion es independiente de alpha_mix.
        sys_h = ActivityAwarePolarSystem(
            k=k, n=n, ae_model=hybrid_aes[rho_db],
            p_empty=p_empty, thresh=0.5, alpha_mix=float(alpha_mix),
        )
        print(f'  [Hibrido] Recogiendo {total:,} muestras ...', end=' ', flush=True)
        p_h, a_h = collect_p_active(sys_h, ebno_db, n_batches, batch_size)
        print(f'OK  (vacios: {(~a_h).sum():,}  activos: {a_h.sum():,})')
        results[(rho_db, 'Hybrid')] = sweep_thresholds(p_h, a_h, thresholds)

        # E2E
        enc, dec = e2e_models[rho_db]
        sys_e = E2ESystem(
            k=k, n=n, encoder=enc, decoder=dec,
            p_empty=p_empty, thresh=0.5,
        )
        print(f'  [E2E]    Recogiendo {total:,} muestras ...', end=' ', flush=True)
        p_e, a_e = collect_p_active(sys_e, ebno_db, n_batches, batch_size)
        print(f'OK  (vacios: {(~a_e).sum():,}  activos: {a_e.sum():,})')
        results[(rho_db, 'E2E')] = sweep_thresholds(p_e, a_e, thresholds)

    # ── Figura y CSV ─────────────────────────────────────────────────────────
    _plot_figure(results, thresholds, rho_dbs, fig_dir, show)
    _save_csv(results, thresholds)

    return results


# ---------------------------------------------------------------------------
# Figura de publicacion
# ---------------------------------------------------------------------------

def _plot_figure(
    results: dict,
    thresholds: np.ndarray,
    rho_dbs: tuple,
    fig_dir: str,
    show: bool,
) -> None:
    """
    Genera la figura con dos subplots (uno por SNR):
      - Eje X: tau (lineal, [0, 1])
      - Eje Y: probabilidad (log)
      - Curvas: P_FA y P_MD para Hibrido y E2E
      - Linea vertical en tau=0.5 (valor por defecto del experimento principal)
      - Grid mayor y menor con alpha=0.3 y linestyle='--'
    """
    legend_elems = [
        Line2D([0], [0], color=C_HYBRID, ls='-',  marker='o', ms=4,
               label='Hybrid $P_{FA}$'),
        Line2D([0], [0], color=C_HYBRID, ls='--', marker='s', ms=4,
               label='Hybrid $P_{MD}$'),
        Line2D([0], [0], color=C_E2E,    ls='-',  marker='o', ms=4,
               label='E2E $P_{FA}$'),
        Line2D([0], [0], color=C_E2E,    ls='--', marker='s', ms=4,
               label='E2E $P_{MD}$'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    for ax, rho_db in zip(axes, rho_dbs):
        pfas_h, pmds_h = results[(rho_db, 'Hybrid')]
        pfas_e, pmds_e = results[(rho_db, 'E2E')]

        ax.plot(thresholds, pfas_h, color=C_HYBRID, ls='-',  marker='o', ms=4.5)
        ax.plot(thresholds, pmds_h, color=C_HYBRID, ls='--', marker='s', ms=4.5)
        ax.plot(thresholds, pfas_e, color=C_E2E,    ls='-',  marker='o', ms=4.5)
        ax.plot(thresholds, pmds_e, color=C_E2E,    ls='--', marker='s', ms=4.5)

        # Linea vertical tau=0.5 (valor elegido para el experimento principal)
        ax.axvline(0.5, color='#555555', lw=1.0, ls=':', alpha=0.75, zorder=0)

        # Etiqueta con transformacion mixta (X en coordenadas de datos, Y en ejes)
        # Esto fija la anotacion justo a la derecha de tau=0.5
        # independientemente del rango del eje Y logaritmico.
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(
            0.515, 0.04, L_TAU05,
            transform=trans,
            color='#555555', fontsize=9,
            va='bottom', ha='left', style='italic',
        )

        ax.set_yscale('log')
        ax.set_xlim(0.0, 1.0)

        # Ticks: mayor cada 0.2, menor cada 0.05
        ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.05))

        # Grid mayor y menor: gris muy claro, alpha=0.3, linestyle='--'
        ax.grid(True, which='major', color='#bbbbbb', ls='--', alpha=0.3, lw=0.6)
        ax.grid(True, which='minor', color='#bbbbbb', ls='--', alpha=0.3, lw=0.4)

        ax.set_xlabel(L_TAU_XLABEL)
        ax.set_title(L_RHO.get(rho_db, f'rho={rho_db} dB'))

    axes[0].set_ylabel(L_PROB_YLABEL)
    axes[0].legend(handles=legend_elems, loc='upper left', fontsize=9, framealpha=0.88)

    fig.suptitle(L_SUPTITLE, fontsize=12, y=1.02)
    plt.tight_layout()

    fname_pdf = os.path.join(fig_dir, 'tau_sweep_hybrid_e2e.pdf')
    fname_png = os.path.join(fig_dir, 'tau_sweep_hybrid_e2e.png')
    fig.savefig(fname_pdf)
    fig.savefig(fname_png, dpi=300)
    print(f'\nFigura guardada:\n  {fname_pdf}\n  {fname_png}')

    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Persistencia de resultados
# ---------------------------------------------------------------------------

def _save_csv(results: dict, thresholds: np.ndarray) -> None:
    csv_path   = os.path.join(TABLE_DIR, 'tau_sweep.csv')
    fieldnames = ['rho_db', 'sistema', 'tau', 'p_fa', 'p_md']
    os.makedirs(TABLE_DIR, exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (rho_db, sistema), (pfas, pmds) in results.items():
            for tau, p_fa, p_md in zip(thresholds, pfas, pmds):
                writer.writerow({
                    'rho_db':  rho_db,
                    'sistema': sistema,
                    'tau':     round(float(tau), 4),
                    'p_fa':    '' if np.isnan(p_fa) else round(float(p_fa), 6),
                    'p_md':    '' if np.isnan(p_md) else round(float(p_md), 6),
                })
    print(f'Resultados guardados: {csv_path}')
