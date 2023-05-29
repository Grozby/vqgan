from typing import Dict

import tensorflow as tf


class Discriminator(tf.keras.Model):
    """
    Discriminator from PatchGAN.

    References:
        https://arxiv.org/pdf/1611.07004.pdf
    """

    def __init__(
        self,
        start_channels: int,
        n_layers: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.start_channels = start_channels
        self.n_layers = n_layers

        self.input_convolution()

        self.blocks = [
            tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(
                    filters=start_channels,
                    kernel_size=4,
                    strides=2,
                    padding="same",
                    kernel_initializer=tf.keras.initializers.RandomNormal(
                        mean=0.0,
                        stddev=0.02,
                    ),
                ),
                tf.keras.layers.LeakyReLU(alpha=0.2),
            ])
        ]
        channels_multipliers = [min(2**i, 8) for i in range(n_layers)]

        for i, c_mul in channels_multipliers:
            self.blocks.append(
                tf.keras.models.Sequential([
                    tf.keras.layers.Conv2D(
                        filters=start_channels * c_mul,
                        kernel_size=4,
                        strides=2 if i < n_layers - 1 else 1,
                        padding="same",
                        use_bias=False,
                        kernel_initializer=tf.keras.initializers.RandomNormal(
                            mean=0.0,
                            stddev=0.02,
                        ),
                    ),
                    tf.keras.layers.BatchNormalization(
                        gamma_initializer=tf.keras.initializers.RandomNormal(
                            mean=1.0,
                            stddev=0.02,
                        ),
                        beta_initializer=tf.keras.initializers.Constant(0),
                    ),
                    tf.keras.layers.LeakyReLU(alpha=0.2),
                ]))

        self.output_convolution = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=4,
            strides=1,
            padding="same",
        )

    def call(self, x, training=None, mask=None):
        for block in self.blocks:
            x = block(x)
        return self.output_convolution(x)

    def get_config(self) -> Dict:
        config = super().get_config()
        config.update({
            "start_channels": self.start_channels,
            "n_layers": self.n_layers,
        })
        return config
