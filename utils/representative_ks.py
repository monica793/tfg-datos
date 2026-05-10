"""
Selección de k representativos para barridos comparables híbrido vs E2E.
Solo importa Sionna dentro de las funciones que lo necesitan.
"""

from __future__ import annotations

import numpy as np

K_CAND_DEFAULT = list(range(5, 101, 5))


def k_is_valid_for_5g(k: int, n: int) -> bool:
    from sionna.phy.fec.polar import Polar5GEncoder

    try:
        Polar5GEncoder(k=int(k), n=int(n))
        return True
    except Exception:
        return False


def pick_representative_ks(n: int, k_cand: list[int] | None = None, n_pick: int = 5) -> list[int]:
    """
    Hasta `n_pick` valores de k espaciados entre los válidos para Polar5G(n).
    """
    if k_cand is None:
        k_cand = K_CAND_DEFAULT
    valid = [k for k in k_cand if k < n and k_is_valid_for_5g(k, n)]
    if not valid:
        return []
    if len(valid) <= n_pick:
        return valid
    idx = np.unique(np.round(np.linspace(0, len(valid) - 1, n_pick)).astype(int))
    return [valid[i] for i in idx]
