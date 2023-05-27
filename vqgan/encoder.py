from typing import List, Dict

import tensorflow as tf

from vqgan.layers.attention_block import AttentionBlock
from vqgan.layers.resnet_block import ResNetBlock


class Encoder(tf.keras.Model):

    def __init__(
        self,
        input_resolution: int,
        start_channels: int,
        channel_multipliers: List[int],
        attention_at_resolution: List[int],
        n_blocks: int,
        z_channels: int,
        double_z_channels: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.input_resolution = input_resolution
        self.start_channels = start_channels
        self.channel_multipliers = channel_multipliers
        self.attention_at_resolution = attention_at_resolution
        self.n_blocks = n_blocks
        self.z_channels = z_channels
        self.double_z_channels = double_z_channels

        self.input_conv = tf.keras.layers.Conv2D(
            filters=start_channels,
            kernel_size=3,
            strides=1,
            padding="same",
        )
        self.blocks = []

        current_resolution = input_resolution
        for i, c_mul in enumerate(channel_multipliers):
            in_channels = start_channels * (1 if i == 0 else
                                            channel_multipliers[i - 1])
            out_channels = start_channels * c_mul

            block_layers = []
            for _ in range(n_blocks):
                block_layers.append(
                    ResNetBlock(
                        input_channels=in_channels,
                        output_channels=out_channels,
                    ))
                in_channels = out_channels

                if current_resolution in attention_at_resolution:
                    block_layers.append(AttentionBlock(channels=out_channels))

            if i != len(channel_multipliers) - 1:
                current_resolution //= 2
                block_layers.append(
                    tf.keras.layers.Conv2D(
                        filters=out_channels,
                        kernel_size=3,
                        strides=2,
                        padding="same",
                    ))
            self.blocks.append(tf.keras.models.Sequential(*block_layers))

        self.residual_1 = ResNetBlock(
            input_channels=out_channels,
            output_channels=out_channels,
        )
        self.attention = AttentionBlock(channels=out_channels)
        self.residual_2 = ResNetBlock(
            input_channels=out_channels,
            output_channels=out_channels,
        )
        self.normalize = tf.keras.layers.GroupNormalization(
            groups=32,
            epsilon=1e-6,
        )
        self.output_convolution = tf.keras.layers.Conv2D(
            filters=z_channels * 2 if double_z_channels else z_channels,
            kernel_size=3,
            strides=1,
            padding="same",
        )

    def call(self, x, training=None, mask=None):
        x = self.input_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.residual_1(x)
        x = self.attention(x)
        x = self.residual_2(x)
        x = self.normalize(x)
        x = tf.keras.activations.swish(x)
        return self.output_convolution(x)

    def get_config(self) -> Dict:
        config = super().get_config()
        config.update({
            "input_resolution": self.input_resolution,
            "start_channels": self.start_channels,
            "channel_multipliers": self.channel_multipliers,
            "attention_at_resolution": self.attention_at_resolution,
            "n_blocks": self.n_blocks,
            "z_channels": self.z_channels,
            "double_z_channels": self.double_z_channels,
        })
        return config
