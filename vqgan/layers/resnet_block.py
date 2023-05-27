from typing import Dict

import tensorflow as tf


class ResNetBlock(tf.keras.layers.Layer):

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.group_norm_1 = tf.keras.layers.GroupNormalization(
            groups=32,
            epsilon=1e-6,
        )
        self.group_norm_2 = tf.keras.layers.GroupNormalization(
            groups=32,
            epsilon=1e-6,
        )
        self.convolution_1 = tf.keras.layers.Conv2D(
            filters=output_channels,
            kernel_size=3,
            strides=1,
            padding="same",
        )
        self.convolution_2 = tf.keras.layers.Conv2D(
            filters=output_channels,
            kernel_size=3,
            strides=1,
            padding="same",
        )
        if input_channels != output_channels:
            self.shortcut = tf.keras.layers.Conv2D(
                filters=output_channels,
                kernel_size=3,
                strides=1,
                padding="same",
            )
        else:
            self.shortcut = tf.keras.layers.Identity()

    def call(self, x, *args):
        previous = x

        x = self.group_norm_1(x)
        x = tf.keras.activations.swish(x)
        x = self.convolution_1(x)

        x = self.group_norm_2(x)
        x = tf.keras.activations.swish(x)
        x = self.convolution_2(x)

        return x + self.shortcut(previous)

    def get_config(self) -> Dict:
        config = super().get_config()
        config.update({
            "input_channels": self.input_channels,
            "output_channels": self.output_channels
        })
        return config
