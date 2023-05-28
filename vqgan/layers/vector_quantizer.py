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
        distances = (tf.math.reduce_sum(
            z_flatten**2,
            axis=1,
            keepdims=True,
        ) + tf.math.reduce_sum(
            self.embedding.weights**2,
            axis=0,
            keepdims=True,
        ) - (2 * z_flatten @ tf.transpose(self.embedding.weights)))

        encoding_indices = tf.argmin(distances, axis=1)
        quantized = self.embedding(encoding_indices).reshape(z.shape)

        vq_loss = tf.math.reduce_mean((quantized - tf.stop_gradient(z))**2)
        commitment_loss = (self.beta * tf.math.reduce_mean(
            (tf.stop_gradient(quantized) - z)**2))

        quantized = z + tf.stop_gradient(quantized - z)

        encodings = tf.one_hot(encoding_indices,
                               self.num_embeddings,
                               dtype=distances.dtype)
        avg_probs = tf.reduce_mean(encodings, 0)
        perplexity = tf.exp(-tf.reduce_sum(avg_probs *
                                           tf.math.log(avg_probs + 1e-10)))

        return quantized, encoding_indices, vq_loss, commitment_loss, perplexity

    def get_config(self) -> Dict:
        config = super().get_config()
        config.update({
            "number_embeddings": self.number_embeddings,
            "embedding_dimension": self.embedding_dimension,
            "beta": self.beta,
        })
        return config
