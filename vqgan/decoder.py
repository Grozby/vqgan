from typing import List, Dict

import tensorflow as tf

from vqgan.layers.attention_block import AttentionBlock
from vqgan.layers.resnet_block import ResNetBlock
from vqgan.layers.upsampling import UpSampling


class Decoder(tf.keras.Model):

    def __init__(
        self,
        input_resolution: int,
        start_channels: int,
        output_channels: int,
        channel_multipliers: List[int],
        attention_at_resolution: List[int],
        n_blocks: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.input_resolution = input_resolution
        self.start_channels = start_channels
        self.output_channels = output_channels
        self.channel_multipliers = channel_multipliers
        self.attention_at_resolution = attention_at_resolution
        self.n_blocks = n_blocks

        in_channels = start_channels * channel_multipliers[-1]
        self.input_conv = tf.keras.layers.Conv2D(
            filters=in_channels,
            kernel_size=3,
            strides=1,
            padding="same",
        )

        self.residual_1 = ResNetBlock(
            input_channels=in_channels,
            output_channels=in_channels,
        )
        self.attention = AttentionBlock(channels=in_channels)
        self.residual_2 = ResNetBlock(
            input_channels=in_channels,
            output_channels=in_channels,
        )

        self.blocks = []

        current_resolution = (input_resolution //
                              2**(len(channel_multipliers) - 1))
        for i, c_mul in enumerate(reversed(channel_multipliers)):
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
                current_resolution *= 2
                block_layers.append(UpSampling(channels=out_channels))
            self.blocks.append(tf.keras.models.Sequential(block_layers))

        self.normalize = tf.keras.layers.GroupNormalization(
            groups=32,
            epsilon=1e-6,
        )
        self.output_convolution = tf.keras.layers.Conv2D(
            filters=output_channels,
            kernel_size=3,
            strides=1,
            padding="same",
        )

    def call(self, x, training=None, **kwargs):
        x = self.input_conv(x)
        x = self.residual_1(x, training=None)
        x = self.attention(x, training=None)
        x = self.residual_2(x, training=None)
        for block in self.blocks:
            x = block(x, training=None)
        x = self.normalize(x)
        x = tf.keras.activations.swish(x)
        return self.output_convolution(x)

    def get_config(self) -> Dict:
        config = super().get_config()
        config.update({
            "input_resolution": self.input_resolution,
            "start_channels": self.start_channels,
            "output_channels": self.output_channels,
            "channel_multipliers": self.channel_multipliers,
            "attention_at_resolution": self.attention_at_resolution,
            "n_blocks": self.n_blocks,
        })
        return config
