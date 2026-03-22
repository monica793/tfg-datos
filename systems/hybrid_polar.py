import tensorflow as tf
from sionna.phy.mapping import Constellation, Mapper, Demapper, BinarySource
from sionna.phy.fec.polar import Polar5GEncoder, Polar5GDecoder
from sionna.phy.utils import ebnodb2no
from sionna.phy.channel import AWGN

from utils.signal import c2ri, ri2c


class ActivityAwarePolarSystem(tf.keras.Model):
    """
    Pipeline híbrido completo: Polar5G + BPSK + SupervisedAE (bloque verde).

    Flujo:
        bits u -> Polar encoder -> BPSK mapper -> [* actividad] -> AWGN
        -> SupervisedAE (denoise + detectar actividad)
        -> Demapper -> Polar decoder -> u_hat

    Devuelve: u, u_hat, c_true, c_hat, a_true, p_active, a_hat
    """

    def __init__(self, k, n, ae_model, p_empty=0.3, thresh=0.5):
        super().__init__()
        self.k = k
        self.n = n
        self.rate = k / n
        self.p_empty = float(p_empty)
        self.thresh = float(thresh)
        self.ae = ae_model

        self.source = BinarySource()
        self.encoder = Polar5GEncoder(k=k, n=n)
        self.const = Constellation("pam", num_bits_per_symbol=1, trainable=False)
        self.mapper = Mapper(constellation=self.const)
        self.channel = AWGN()
        self.demapper = Demapper("app", constellation=self.const)
        self.decoder = Polar5GDecoder(self.encoder, dec_type="SCL", list_size=8)

    @tf.function
    def call(self, inputs, training=False):
        batch_size, ebno_db = inputs
        batch_size = tf.cast(batch_size, tf.int32)

        # Actividad aleatoria: 1 = canal activo, 0 = silencio
        a_true = tf.cast(tf.random.uniform([batch_size, 1]) > self.p_empty, tf.float32)

        # TX
        u = self.source([batch_size, self.k])
        c_true = self.encoder(u) # guardamos el codeword real
        x_info = self.mapper(c_true)
        x = x_info * tf.cast(a_true, x_info.dtype) # silencio si a_true=0

        # Canal AWGN
        no = ebnodb2no(ebno_db, num_bits_per_symbol=1, coderate=self.rate)
        y = self.channel(x, no)

        # Bloque verde: SupervisedAE
        y_ri = c2ri(y)
        x_hat_ri, p_active = self.ae(y_ri, training=training)
        y_hat = ri2c(x_hat_ri)
        a_hat = tf.cast(p_active > self.thresh, tf.float32)

        # Decodificación solo para bloques detectados como activos
        idx = tf.squeeze(tf.where(tf.squeeze(a_hat, axis=-1) > 0.5), axis=1)

        u_hat_full = tf.zeros([batch_size, self.k], dtype=tf.float32)
        c_hat_full = tf.zeros([batch_size, self.n], dtype=tf.float32) #codeword estimado

        def decode_selected():
            y_sel = tf.gather(y_hat, idx)      # [Bsel,N]
            llr_sel = self.demapper(y_sel, no) # [Bsel,N]
            u_hat_sel = self.decoder(llr_sel)  # [Bsel,K]
            u_hat_sel = tf.clip_by_value(tf.round(tf.cast(u_hat_sel, tf.float32)), 0.0, 1.0)
            # Re-encode para obtener c_hat_sel. 
            c_hat_sel = self.encoder(u_hat_sel)  # [Bsel,N]
            c_hat_sel = tf.clip_by_value(tf.round(tf.cast(c_hat_sel, tf.float32)), 0.0, 1.0)
            u_upd = tf.tensor_scatter_nd_update(
                u_hat_full, tf.expand_dims(idx, axis=1), u_hat_sel)
            c_upd = tf.tensor_scatter_nd_update(
                c_hat_full, tf.expand_dims(idx, axis=1), c_hat_sel)
            return u_upd, c_upd

        u_hat_full, c_hat_full = tf.cond(
            tf.size(idx) > 0,
            decode_selected,
            lambda: (u_hat_full, c_hat_full)
        )

        return u, u_hat_full, c_true, c_hat_full, a_true, p_active, a_hat