from typing import Dict

import tensorflow as tf

from vqgan.discriminator import Discriminator
from vqgan.losses.lpips import LPIPS
from vqgan.vqgan import VQGAN


class TrainVQGAN(tf.keras.Model):

    def __init__(
        self,
        *args,
        vqgan_kwargs: Dict,
        discriminator_kwargs: Dict,
        lpips_weights_path: str,
        discriminator_step_start: int,
        perceptual_loss_factor: float = 1.,
        reconstruction_loss_factor: float = 1.,
        discriminator_loss_factor: float = 1.,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.vqgan_kwargs = vqgan_kwargs
        self.discriminator_kwargs = discriminator_kwargs
        self.discriminator_step_start = discriminator_step_start
        self.perceptual_loss_factor = perceptual_loss_factor
        self.reconstruction_loss_factor = reconstruction_loss_factor
        self._discriminator_loss_factor = discriminator_loss_factor

        self.vqgan = VQGAN(**vqgan_kwargs)
        self.discriminator = Discriminator(**discriminator_kwargs)
        self.perceptual_loss = LPIPS(lpips_weights_path)

        # Losses for VQVAE
        self.loss_reconstruction_tracker = tf.keras.metrics.Mean(
            name="loss_reconstruction")
        self.loss_perceptual_tracker = tf.keras.metrics.Mean(name="loss_lpips")
        self.loss_codebook_tracker = tf.keras.metrics.Mean(name="loss_vq")
        self.loss_generator_tracker = tf.keras.metrics.Mean(
            name="loss_generator")
        self.loss_vqgan_tracker = tf.keras.metrics.Mean(name="loss_vqgan")
        # Loss for Discriminator
        self.loss_discriminator_tracker = tf.keras.metrics.Mean(
            name="loss_discriminator")

        self._step = tf.Variable(0, dtype=tf.int64)

        self.optimizer_vqgan = None
        self.optimizer_discriminator = None

    @property
    def metrics(self):
        return [
            self.loss_reconstruction_tracker,
            self.loss_perceptual_tracker,
            self.loss_codebook_tracker,
            self.loss_generator_tracker,
            self.loss_vqgan_tracker,
            self.loss_discriminator_tracker,
        ]

    def compile(
        self,
        *args,
        optimizer_vqgan,
        optimizer_discriminator,
        **kwargs,
    ):
        super().compile()
        self.optimizer_vqgan = optimizer_vqgan
        self.optimizer_discriminator = optimizer_discriminator

    @property
    def discriminator_loss_factor(self) -> float:
        return tf.where(self._step.value() > self.discriminator_step_start,
                        self._discriminator_loss_factor, 0.)

    def _calculate_lambda(self, perceptual_loss, gan_loss):
        last_layer_weights = self.vqgan.decoder.output_convolution.weights
        perceptual_loss_grads = tf.gradients(
            perceptual_loss,
            last_layer_weights,
        )[0]
        gan_loss_grads = tf.gradients(
            gan_loss,
            last_layer_weights,
        )[0]

        lambda_value = tf.norm(perceptual_loss_grads)
        lambda_value /= (tf.norm(gan_loss_grads) + 1e-4)
        lambda_value = tf.clip_by_value(
            lambda_value,
            clip_value_min=0,
            clip_value_max=1e4,
        )
        return 0.8 * lambda_value

    def train_step(self, images):
        self._step.assign_add(1)
        with tf.GradientTape() as tape:
            # Run VQVAE
            decoded_images, _, q_loss = self.vqgan(images)

            # Run Discriminator
            discriminator_fake = self.discriminator(decoded_images)

            # Run LPIPS
            perceptual_loss = self.perceptual_loss([
                images,
                decoded_images,
            ]) * self.perceptual_loss_factor
            reconstruction_loss = (tf.abs(images - decoded_images) *
                                   self.reconstruction_loss_factor)

            pr_loss = tf.reduce_mean(perceptual_loss + reconstruction_loss)
            generator_loss = -tf.reduce_mean(discriminator_fake)

            lambda_value = self._calculate_lambda(pr_loss, generator_loss)
            generator_loss *= self.discriminator_loss_factor * lambda_value
            vq_loss = pr_loss + q_loss + generator_loss

        vqgan_gradients = tape.gradient(
            vq_loss,
            self.vqgan.trainable_variables,
        )
        self.optimizer_vqgan.apply_gradients(
            zip(
                vqgan_gradients,
                self.vqgan.trainable_variables,
            ))

        with tf.GradientTape() as tape:
            # Run Discriminator
            discriminator_real = self.discriminator(images)
            discriminator_fake = self.discriminator(decoded_images)

            d_loss_real = tf.reduce_mean(tf.nn.relu(1. - discriminator_real))
            d_loss_fake = tf.reduce_mean(tf.nn.relu(1. + discriminator_fake))
            discriminator_loss = (self.discriminator_loss_factor * 0.5 *
                                  (d_loss_real + d_loss_fake))

        discriminator_gradients = tape.gradient(
            discriminator_loss,
            self.discriminator.trainable_variables,
        )
        self.optimizer_vqgan.apply_gradients(
            zip(
                discriminator_gradients,
                self.discriminator.trainable_variables,
            ))

        self.loss_reconstruction_tracker.update_state(reconstruction_loss)
        self.loss_perceptual_tracker.update(perceptual_loss)
        self.loss_codebook_tracker.update(q_loss)
        self.loss_generator_tracker.update(generator_loss)
        self.loss_vqgan_tracker.update_state(vq_loss)
        self.loss_discriminator_tracker.update_state(discriminator_loss)

        loss_vqgan_result = self.loss_vqgan_tracker.result()
        loss_discriminator_result = self.loss_discriminator_tracker.result()
        return {
            "loss": loss_vqgan_result + loss_discriminator_result,
            "loss_vqgan": loss_vqgan_result,
            "loss_discriminator": loss_discriminator_result,
        }
