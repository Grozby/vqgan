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
            padding=0,
        )
        self.k = tf.keras.layers.Conv2D(
            filters=channels,
            kernel_size=1,
            strides=1,
            padding=0,
        )
        self.v = tf.keras.layers.Conv2D(
            filters=channels,
            kernel_size=1,
            strides=1,
            padding=0,
        )

        self.linear = tf.keras.layers.Conv2D(
            filters=channels,
            kernel_size=1,
            strides=1,
            padding=0,
        )

    def call(self, x, *args):
        b, h, w, c = x.shape
        previous = x

        x = self.normalize(x)
        q, k, v = self.q(x), self.k(x), self.v(x)
        q, k, v = (
            q.reshape(b, c, h * w),
            k.reshape(b, c, h * w),
            v.reshape(b, c, h * w),
        )

        q = tf.transpose(q, perm=(0, 2, 1))
        x = q @ k * (c**-0.5)  # shape = (b, hw, hw)
        x = tf.nn.softmax(x, axis=-1)
        x = tf.transpose(x, perm=(0, 2, 1))
        x = x @ v  # shape = (b, c, hw)
        x = x.reshape(b, h, w, c)

        return self.linear(x) + previous

    def get_config(self) -> Dict:
        config = super().get_config()
        config.update({
            "channels": self.channels,
        })
        return config
