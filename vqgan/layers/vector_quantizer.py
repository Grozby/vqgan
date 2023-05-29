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


class VectorQuantizerEMA(tf.keras.layers.Layer):

    def __init__(
        self,
        number_embeddings: int,
        embedding_dimension: int,
        beta: float,
        decay: float,
        epsilon: float = 1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.number_embeddings = number_embeddings
        self.embedding_dimension = embedding_dimension
        self.beta = beta
        self.decay = decay
        self.epsilon = epsilon

        self.embedding = tf.keras.layers.Embedding(
            input_dim=number_embeddings,
            output_dim=embedding_dimension,
            embeddings_initializer=tf.keras.initializers.RandomUniform(
                minval=-1 / number_embeddings,
                maxval=1 / number_embeddings,
            ),
        )

        self.ema_cluster_size = tf.train.ExponentialMovingAverage(
            decay=self.decay,
            name='ema_cluster_size',
        )
        self.ema_cluster_size.initialize(
            tf.zeros(
                [number_embeddings],
                dtype=tf.float32,
            ))

        self.ema_dw = tf.train.ExponentialMovingAverage(
            decay=self.decay,
            name='ema_dw',
        )
        self.ema_dw.initialize(self.embeddings.weights)

    def call(self, z, training=None, *args):
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
        encodings = tf.one_hot(
            encoding_indices,
            self.num_embeddings,
            dtype=distances.dtype,
        )

        if training:
            sum_encodings = tf.reduce_sum(encodings, axis=0)
            dw = tf.matmul(z_flatten, encodings, transpose_a=True)

            self.ema_cluster_size.apply(sum_encodings)
            updated_ema_cluster_size = self.ema_cluster_size.average(
                sum_encodings)
            self.ema_dw.apply(dw)
            updated_ema_dw = self.ema_dw.average(dw)

            n = tf.reduce_sum(updated_ema_cluster_size)
            updated_ema_cluster_size = (
                (updated_ema_cluster_size + self.epsilon) /
                (n + self.num_embeddings * self.epsilon) * n)

            normalised_updated_ema_w = (updated_ema_dw / tf.reshape(
                updated_ema_cluster_size,
                (1, -1),
            ))
            self.embeddings.weights.assign(normalised_updated_ema_w)

        commitment_loss = (self.beta * tf.math.reduce_mean(
            (tf.stop_gradient(quantized) - z)**2))

        quantized = z + tf.stop_gradient(quantized - z)

        avg_probs = tf.reduce_mean(encodings, 0)
        perplexity = tf.exp(-tf.reduce_sum(avg_probs *
                                           tf.math.log(avg_probs + 1e-10)))

        return quantized, encoding_indices, 0, commitment_loss, perplexity

    def get_config(self) -> Dict:
        config = super().get_config()
        config.update({
            "number_embeddings": self.number_embeddings,
            "embedding_dimension": self.embedding_dimension,
            "beta": self.beta,
            "decay": self.decay,
            "epsilon": self.epsilon,
        })
        return config
