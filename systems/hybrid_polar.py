import numpy as np
import torch
import tensorflow as tf

from sionna.phy.mapping import Constellation, Mapper, Demapper, BinarySource
from sionna.phy.fec.polar import Polar5GEncoder, Polar5GDecoder
from sionna.phy.utils import ebnodb2no
from sionna.phy.channel import AWGN

from utils.signal import c2ri, ri2c


class ActivityAwarePolarSystem:
    """
    Pipeline híbrido completo: Polar5G + BPSK + SupervisedAE.

    Flujo:
        bits u -> Polar encoder -> BPSK mapper -> [* actividad] -> AWGN
        -> SupervisedAE (denoise + detectar actividad)
        -> Demapper -> Polar decoder -> u_hat

    Devuelve: u, u_hat, c_true, c_hat, a_true, p_active, a_hat
    Todo en numpy para ser agnóstico entre PyTorch y TensorFlow.
    """

    def __init__(self, k, n, ae_model, p_empty=0.3, thresh=0.5):
        self.k = k
        self.n = n
        self.rate = k / n
        self.p_empty = float(p_empty)
        self.thresh = float(thresh)
        self.ae = ae_model

        self.source  = BinarySource()
        self.encoder = Polar5GEncoder(k=k, n=n)
        self.const   = Constellation("pam", num_bits_per_symbol=1, trainable=False)
        self.mapper  = Mapper(constellation=self.const)
        self.channel = AWGN()
        self.demapper = Demapper("app", constellation=self.const)
        self.decoder  = Polar5GDecoder(self.encoder, dec_type="SCL", list_size=8)

    def __call__(self, batch_size, ebno_db):
        """
        batch_size: entero Python
        ebno_db:    float Python
        """
        # Actividad: 1=activo, 0=silencio
        a_np = (np.random.rand(batch_size, 1) > self.p_empty).astype(np.float32)

        # TX — Sionna 2.0 devuelve tensores PyTorch
        u_pt     = self.source([batch_size, self.k])
        c_pt     = self.encoder(u_pt)
        x_info_pt = self.mapper(c_pt)

        # Convertir a numpy complejo
        x_info_np = x_info_pt.detach().cpu().numpy()
        x_np = x_info_np * a_np  # silencio si a=0

        # Canal AWGN en numpy
        no_val = 10 ** (-ebno_db / 10) / self.rate
        noise  = np.sqrt(no_val / 2) * (
            np.random.randn(*x_np.shape) + 1j * np.random.randn(*x_np.shape)
        )
        y_np = x_np + noise.astype(x_np.dtype)

        # Bloque verde: SupervisedAE (TensorFlow)
        y_tf = tf.complex(
            tf.constant(y_np.real, dtype=tf.float32),
            tf.constant(y_np.imag, dtype=tf.float32)
        )
        y_ri = c2ri(y_tf)
        x_hat_ri, p_active_tf = self.ae(y_ri, training=False)
        p_active_np = p_active_tf.numpy()           # [B, 1]
        a_hat_np    = (p_active_np > self.thresh).astype(np.float32)

        # Reconstruir señal limpia desde AE
        y_hat_tf = ri2c(x_hat_ri)
        y_hat_np = y_hat_tf.numpy()

        # Decodificación solo en bloques detectados como activos
        active_idx = np.where(a_hat_np.squeeze() > 0.5)[0]

        u_np     = u_pt.detach().cpu().numpy()
        c_true_np = c_pt.detach().cpu().numpy()
        u_hat_np  = np.zeros((batch_size, self.k), dtype=np.float32)
        c_hat_np  = np.zeros((batch_size, self.n), dtype=np.float32)

        if len(active_idx) > 0:
            y_sel_np  = y_hat_np[active_idx]
            y_sel_pt  = torch.tensor(y_sel_np)

            no_pt    = torch.tensor(float(no_val))
            llr_pt   = self.demapper(y_sel_pt, no_pt)
            u_hat_pt = self.decoder(llr_pt)
            u_hat_sel = np.clip(np.round(u_hat_pt.detach().cpu().numpy()), 0, 1).astype(np.float32)

            # Re-encode para obtener c_hat
            u_hat_pt2 = torch.tensor(u_hat_sel.astype(np.float32))
            c_hat_sel = np.clip(
                np.round(self.encoder(u_hat_pt2).detach().cpu().numpy()), 0, 1
            ).astype(np.float32)

            u_hat_np[active_idx] = u_hat_sel
            c_hat_np[active_idx] = c_hat_sel

        return u_np, u_hat_np, c_true_np, c_hat_np, a_np, p_active_np, a_hat_np