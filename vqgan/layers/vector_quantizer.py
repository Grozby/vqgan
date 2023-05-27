from typing import Dict

import tensorflow as tf


class VectorQuantizer(tf.keras.layers.Layer):

    def __init__(
        self,
        number_embeddings: int,
        embedding_dimension: int,
        beta: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.number_embeddings = number_embeddings
        self.embedding_dimension = embedding_dimension
        self.beta = beta

        self.embedding = tf.keras.layers.Embedding(
            input_dim=number_embeddings,
            output_dim=embedding_dimension,
            embeddings_initializer=tf.keras.initializers.RandomUniform(
                minval=-1 / number_embeddings,
                maxval=1 / number_embeddings,
            ),
        )

    def call(self, z, *args):
        z_flatten = tf.reshape(z, (-1, self.embedding_dimension))
        d = (tf.math.reduce_sum(z_flatten**2, axis=1, keepdims=True) +
             tf.math.reduce_sum(self.embedding.weights**2, axis=1) -
             (2 * z_flatten @ tf.transpose(self.embedding.weights)))
        min_encoding_indices = tf.argmin(d, axis=1)
        z_q = self.embedding(min_encoding_indices).reshape(z.shape)

        loss = tf.math.reduce_mean((tf.stop_gradient(z_q) - z)**2)
        loss += self.beta * tf.math.reduce_mean((z_q - tf.stop_gradient(z))**2)

        z_q = z + tf.stop_gradient(z_q - z)

        return z_q, min_encoding_indices, loss

    def get_config(self) -> Dict:
        config = super().get_config()
        config.update({
            "number_embeddings": self.number_embeddings,
            "embedding_dimension": self.embedding_dimension,
            "beta": self.beta,
        })
        return config
