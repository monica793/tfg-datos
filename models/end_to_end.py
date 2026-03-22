import torch
import torch.nn as nn


# ============================================================
# STE: binarizador diferenciable (mismo truco que el TurboAE)
# Forward:  redondea a +1 / -1
# Backward: gradiente pasa como si fuera identidad
# ============================================================
class BinarizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.sign()                          # +1 o -1

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out                          # gradiente sin modificar


def binarize(x: torch.Tensor) -> torch.Tensor:
    x = torch.where(x == 0, torch.ones_like(x), x)   # evita 0 exacto
    return BinarizeSTE.apply(x)


# ============================================================
# Encoder: k bits (±1) → n símbolos BPSK (±1)
# ============================================================
class E2EEncoder(nn.Module):
    """
    Aprende a codificar k bits en n símbolos BPSK.
    Equivale a un código de canal aprendido.

    Entrada:  u  [B, k]  bits en {-1, +1}
    Salida:   x  [B, n]  símbolos BPSK en {-1, +1}
    """
    def __init__(self, k: int, n: int, hidden_dim: int = 256):
        super().__init__()
        self.k = k
        self.n = n

        self.net = nn.Sequential(
            nn.Linear(k, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n),
            nn.Tanh(),          # salida continua en (-1, +1)
        )

    def forward(self, u_pm: torch.Tensor) -> torch.Tensor:
        """
        u_pm: [B, k] en {-1, +1}
        Devuelve x: [B, n] en {-1, +1} (binarizado con STE)
        """
        x_cont = self.net(u_pm)       # [B, n] continuo en (-1, +1)
        x      = binarize(x_cont)     # [B, n] binarizado, gradiente pasa
        return x


# ============================================================
# Decoder: n símbolos ruidosos → k bits + detección de actividad
# ============================================================
class E2EDecoder(nn.Module):
    """
    Recibe n símbolos ruidosos y estima:
      - u_hat:    [B, k]  probabilidad de cada bit (rama decodificación)
      - p_active: [B, 1]  probabilidad de canal activo (rama detección)

    Entrada:  y  [B, n]  señal recibida con ruido
    """
    def __init__(self, k: int, n: int, hidden_dim: int = 256):
        super().__init__()
        self.k = k
        self.n = n

        # Tronco compartido
        self.trunk = nn.Sequential(
            nn.Linear(n, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Rama 1: decodificación de bits
        self.head_bits = nn.Sequential(
            nn.Linear(hidden_dim, k),
            nn.Sigmoid(),           # probabilidad de bit=1
        )

        # Rama 2: detección de actividad
        self.head_activity = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),           # probabilidad de canal activo
        )

    def forward(self, y: torch.Tensor):
        """
        y: [B, n]
        Devuelve (u_hat [B,k], p_active [B,1])
        """
        h        = self.trunk(y)
        u_hat    = self.head_bits(h)
        p_active = self.head_activity(h)
        return u_hat, p_active