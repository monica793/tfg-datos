import tensorflow as tf


class SupervisedAE(tf.keras.Model):
    """
    Autoencoder supervisado para denoising y detección de actividad.
    - Reconstruye X (símbolos limpios) desde Y (noisy). 
    - Clasifica entre "activo/vacío" desde el latente Z. 

    Entrada:  y_ri  [B, N, 2]  — señal recibida (Re, Im) con ruido
    Salidas:
        x_hat_ri  [B, N, 2]  — señal reconstruida (rama denoising)
        p_active  [B, 1]     — probabilidad de canal activo (rama clasificador)
    """

    def __init__(self, n, latent_dim=64, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.n = n
        self.in_dim = n * 2  # (I, Q) por símbolo aplanados

        # Encoder: señal ruidosa -> espacio latente z
        self.enc = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation="tanh"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(latent_dim, activation=None), #comprime y no activa para permitir que el espacio latente tenga cualquier rango de valores. 
        ])

        # Decoder: z -> señal reconstruida (Re, Im)
        self.dec = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation="tanh"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(self.in_dim, activation="tanh"), # Fuerza a que la salida reconstruida esté en el rango [-1, 1]. 
        ])

        # Clasificador: z -> probabilidad de canal activo
        self.cls = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation="tanh"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(1, activation="sigmoid"), #Comprime la salida entre cero y uno. 
        ])

    def call(self, y_ri, training=False):
        #y_ri: [B,N,2]
        b = tf.shape(y_ri)[0]
        y_flat = tf.reshape(y_ri, [b, self.in_dim]) # Aplana de  a [Batch, N*2], porque las capas Dense(MLP) solo entienden vectores planos. 
        z = self.enc(y_flat, training=training)
        x_hat_flat = self.dec(z, training=training) # Rama 1: reconstruye señal.  
        p_active = self.cls(z, training=training) #Rama 2: detecta actividad.
        x_hat_ri = tf.reshape(x_hat_flat, [b, self.n, 2]) # Convierte la salida plana de vuelta al formato [Batch, N, 2] para que coincida con la forma de la entrada original (útil para calcular pérdidas de reconstrucción como MSE). 
        return x_hat_ri, p_active

    def encode(self, y_ri, training=False):
        b = tf.shape(y_ri)[0]
        y_flat = tf.reshape(y_ri, [b, self.in_dim])
        return self.enc(y_flat, training=training)