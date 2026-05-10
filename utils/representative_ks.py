"""
Selección de k representativos para barridos (p. ej. E2E con n fijo).

No usa Sionna ni códigos Polar: cualquier entero k con 0 < k < n es válido para el E2E.
Para alinear con el híbrido Polar, filtra después con
`training.train_hybrid.k_is_valid_for_5g` o usa `polar_k_constraint=True` en
`run_curves_for_n`.
"""

from __future__ import annotations

import numpy as np

K_CAND_DEFAULT = list(range(5, 101, 5))


def pick_representative_ks(n: int, k_cand: list[int] | None = None, n_pick: int = 5) -> list[int]:
    """
    Hasta `n_pick` valores de k espaciados dentro de `k_cand`, con k < n.
    """
    if k_cand is None:
        k_cand = K_CAND_DEFAULT
    valid = [k for k in k_cand if 0 < k < n]
    if not valid:
        return []
    if len(valid) <= n_pick:
        return valid
    idx = np.unique(np.round(np.linspace(0, len(valid) - 1, n_pick)).astype(int))
    return [valid[i] for i in idx]
