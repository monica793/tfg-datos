import numpy as np
import torch
from models.end_to_end import E2EEncoder, E2EDecoder


class E2ESystem:
    """
    Pipeline end-to-end completo: E2EEncoder + AWGN + E2EDecoder.

    A diferencia del híbrido, no hay código Polar — el encoder y el decoder
    aprenden juntos la mejor estrategia de codificación/decodificación.

    Flujo:
        bits u (0/1) → ±1 → E2EEncoder → [* actividad] → AWGN
        → E2EDecoder → u_hat (prob.) + p_active

    Devuelve todo en numpy para ser compatible con la evaluación del híbrido.
    """

    def __init__(self, k: int, n: int, encoder: E2EEncoder, decoder: E2EDecoder,
                 p_empty: float = 0.3, thresh: float = 0.5):
        self.k        = k
        self.n        = n
        self.encoder  = encoder
        self.decoder  = decoder
        self.p_empty  = p_empty
        self.thresh   = thresh

    @staticmethod
    def bits01_to_pm1(u: torch.Tensor) -> torch.Tensor:
        """Convierte bits 0/1 a ±1."""
        return u.float() * 2.0 - 1.0

    def __call__(self, batch_size: int, ebno_db: float):
        """
        batch_size: entero Python
        ebno_db:    float Python (Eb/No en dB)

        Devuelve: u, u_hat, a_true, p_active, a_hat  — todo en numpy
        """
        device = next(self.encoder.parameters()).device

        # Actividad aleatoria
        a_np = (np.random.rand(batch_size, 1) > self.p_empty).astype(np.float32)
        a_pt = torch.tensor(a_np, device=device)

        # Bits aleatorios 0/1
        u_np = np.random.randint(0, 2, (batch_size, self.k)).astype(np.float32)
        u_pt = torch.tensor(u_np, device=device)

        with torch.no_grad():
            # TX: bits → símbolos BPSK codificados
            u_pm = self.bits01_to_pm1(u_pt)          # [B, k] en ±1
            x    = self.encoder(u_pm)                 # [B, n] en ±1
            x    = x * a_pt                           # silencio si a=0

            # Canal AWGN
            # Eb/No -> sigma: asumimos potencia unitaria, rate = k/n
            rate  = self.k / self.n
            no_lin = 10 ** (-ebno_db / 10) / rate
            sigma  = np.sqrt(no_lin / 2)
            noise  = torch.randn_like(x) * sigma
            y      = x + noise                        # [B, n]

            # RX: decodificación
            u_hat, p_active = self.decoder(y)         # [B,k], [B,1]

        u_hat_np    = u_hat.cpu().numpy()
        p_active_np = p_active.cpu().numpy()
        a_hat_np    = (p_active_np > self.thresh).astype(np.float32)

        return u_np, u_hat_np, a_np, p_active_np, a_hat_np