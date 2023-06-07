from typing import List

import tensorflow as tf

from vqgan.decoder import Decoder
from vqgan.encoder import Encoder
from vqgan.layers.vector_quantizer import VectorQuantizer


class VQVAE(tf.keras.models.Model):

    def __init__(
        self,
        *,
        input_resolution: int,
        start_channels: int,
        output_channels: int,
        channel_multipliers: List[int],
        attention_at_resolution: List[int],
        n_blocks: int,
        z_channels: int,
        double_z_channels: bool = True,
        number_embeddings: int,
        embedding_dimension: int,
        beta: float,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.encoder = Encoder(
            input_resolution=input_resolution,
            start_channels=start_channels,
            channel_multipliers=channel_multipliers,
            attention_at_resolution=attention_at_resolution,
            n_blocks=n_blocks,
            z_channels=z_channels,
            double_z_channels=double_z_channels,
        )
        self.quantize_convolution = tf.keras.layers.Conv2D(
            filters=embedding_dimension,
            kernel_size=1,
            strides=1,
            padding="valid",
        )
        self.vector_quantize = VectorQuantizer(
            number_embeddings=number_embeddings,
            embedding_dimension=embedding_dimension,
            beta=beta,
        )
        self.post_quantize_convolution = tf.keras.layers.Conv2D(
            filters=embedding_dimension,
            kernel_size=1,
            strides=1,
            padding="valid",
        )
        self.decoder = Decoder(
            input_resolution=input_resolution,
            start_channels=start_channels,
            output_channels=output_channels,
            channel_multipliers=channel_multipliers,
            attention_at_resolution=attention_at_resolution,
            n_blocks=n_blocks,
        )

    def call(self, x, training=None, **kwargs):
        x = self.encoder(x, training=training)
        x = self.quantize_convolution(x, training=training)
        cookbook_entry, cookbook_indexes, vq_loss, commitment_loss, _ = (
            self.vector_quantize(x, training=training))
        x = self.post_quantize_convolution(cookbook_entry, training=training)
        x = self.decoder(x, training=training)
        return x, cookbook_indexes, vq_loss + commitment_loss
