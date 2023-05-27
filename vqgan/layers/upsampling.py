from typing import Dict

import tensorflow as tf


class UpSampling(tf.keras.layers.Layer):

    def __init__(
        self,
        channels: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.channels = channels

        self.up_sample = tf.keras.layers.UpSampling2D(
            size=2,
            interpolation="nearest",
        )
        self.convolution = tf.keras.layers.Conv2D(
            filters=channels,
            kernel_size=3,
            strides=1,
            padding="same",
        )

    def call(self, x, *args):
        x = self.up_sample(x)
        return self.convolution(x)

    def get_config(self) -> Dict:
        config = super().get_config()
        config.update({
            "channels": self.channels,
        })
        return config
