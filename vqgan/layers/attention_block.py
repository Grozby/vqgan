from typing import Dict

import tensorflow as tf


class AttentionBlock(tf.keras.layers.Layer):

    def __init__(
        self,
        channels: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.channels = channels

        self.normalize = tf.keras.layers.GroupNormalization(
            groups=32,
            epsilon=1e-6,
        )
        self.q = tf.keras.layers.Conv2D(
            filters=channels,
            kernel_size=1,
            strides=1,
            padding="valid",
        )
        self.k = tf.keras.layers.Conv2D(
            filters=channels,
            kernel_size=1,
            strides=1,
            padding="valid",
        )
        self.v = tf.keras.layers.Conv2D(
            filters=channels,
            kernel_size=1,
            strides=1,
            padding="valid",
        )

        self.linear = tf.keras.layers.Conv2D(
            filters=channels,
            kernel_size=1,
            strides=1,
            padding="valid",
        )

    def call(self, x, *args):
        previous = x

        x = self.normalize(x)
        q, k, v = self.q(x), self.k(x), self.v(x)
        b, h, w, c = q.shape
        q, k, v = (
            tf.reshape(q, shape=(-1, c, h * w)),
            tf.reshape(k, shape=(-1, c, h * w)),
            tf.reshape(v, shape=(-1, c, h * w)),
        )

        q = tf.transpose(q, perm=(0, 2, 1))
        x = q @ k * (c**-0.5)  # shape = (b, hw, hw)
        x = tf.nn.softmax(x, axis=-1)
        x = tf.transpose(x, perm=(0, 2, 1))
        x = v @ x  # shape = (b, c, hw)
        x = tf.reshape(x, shape=(-1, h, w, c))

        return self.linear(x) + previous

    def get_config(self) -> Dict:
        config = super().get_config()
        config.update({
            "channels": self.channels,
        })
        return config
