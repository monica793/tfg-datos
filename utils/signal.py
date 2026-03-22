import numpy as np
import tensorflow as tf


def c2ri(x_c):
    """Complejo -> tensor real (..., 2) con (Re, Im)."""
    return tf.stack([tf.math.real(x_c), tf.math.imag(x_c)], axis=-1)


def ri2c(x_ri):
    """Tensor real (..., 2) -> complejo."""
    return tf.complex(x_ri[..., 0], x_ri[..., 1])


def rho_db_to_ebno_db(rho_db, R):
    """
    Convierte SNR por símbolo (rho, dB) a Eb/No (dB) dado rate R = k/n.
    Asume BPSK (1 bit/símbolo): EbNo_dB = rho_dB - 10*log10(R)
    """
    R = float(R)
    if R <= 0:
        return -1e9
    return float(rho_db) - 10.0 * np.log10(R)