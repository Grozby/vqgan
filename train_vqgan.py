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
        epochs: int,
        batch_size: int,
        learning_rate_vqgan: float,
        learning_rate_discriminator: float,
        warmup_steps_percentage: float,
        total_number_images: int,
        discriminator_step_start: int,
        perceptual_loss_factor: float = 1.,
        reconstruction_loss_factor: float = 1.,
        discriminator_loss_factor: float = 1.,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.vqgan_kwargs = vqgan_kwargs
        self.discriminator_kwargs = discriminator_kwargs
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate_vqgan = learning_rate_vqgan
        self.learning_rate_discriminator = learning_rate_discriminator
        self.warmup_steps_percentage = warmup_steps_percentage
        self.total_number_images = total_number_images
        self.discriminator_step_start = discriminator_step_start
        self.perceptual_loss_factor = perceptual_loss_factor
        self.reconstruction_loss_factor = reconstruction_loss_factor
        self._discriminator_loss_factor = discriminator_loss_factor

        self.vqgan = VQGAN(**vqgan_kwargs)
        self.discriminator = Discriminator(**discriminator_kwargs)
        self.discriminator.apply(weights_init)
        self.perceptual_loss = LPIPS("vqgan/losses/lpips_weights.h5")

        self.loss_vqgan_tracker, self.loss_discriminator_tracker = (
            tf.keras.metrics.Mean(name="loss_vqgan"),
            tf.keras.metrics.Mean(name="loss_vqgan"),
        )

        self._step = tf.Variable(0, dtype=tf.int64)

        self.optimizer_vqgan = None
        self.opt_discriminator = None

        # total_steps = (total_number_images // batch_size) * epochs
        # warmup_steps = int(warmup_steps_percentage * total_steps)

        # self.optimizer_vqgan = tf.keras.optimizers.Adam(
        #     learning_rate=WarmUpCosineDecay(
        #         learning_rate=learning_rate_vqgan,
        #         warmup_steps=warmup_steps,
        #         total_steps=total_steps,
        #         hold_steps=0,
        #     ),
        #     epsilon=1e-08,
        # )
        # self.opt_discriminator = tf.keras.optimizers.Adam(
        #     learning_rate=learning_rate_discriminator,
        #     epsilon=1e-08,
        # )

    def compile(
        self,
        *args,
        optimizer_vqgan,
        opt_discriminator,
        **kwargs,
    ):
        super().compile()
        self.optimizer_vqgan = optimizer_vqgan
        self.opt_discriminator = opt_discriminator

    @property
    def discriminator_loss_factor(self) -> float:
        return (self._discriminator_loss_factor
                if self._step.value() > self.discriminator_step_start else 0.)

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

        lambda_value = tf.norm(perceptual_loss_grads, ord="fro")
        lambda_value /= (tf.norm(gan_loss_grads, ord="fro") + 1e-4)
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
            discriminator_real = self.discriminator(images)
            discriminator_fake = self.discriminator(decoded_images)

            # Run LPIPS
            perceptual_loss = self.perceptual_loss(
                images,
                decoded_images,
            )
            reconstruction_loss = tf.abs(images - decoded_images)

            pr_loss = tf.reduce_mean(
                self.perceptual_loss_factor * perceptual_loss +
                self.reconstruction_loss_factor * reconstruction_loss)
            g_loss = -tf.reduce_mean(discriminator_fake)

            lambda_value = self._calculate_lambda(pr_loss, g_loss)
            vq_loss = (pr_loss + q_loss +
                       self.discriminator_factor * lambda_value * g_loss)

            d_loss_real = tf.reduce_mean(tf.nn.relu(1. - discriminator_real))
            d_loss_fake = tf.reduce_mean(tf.nn.relu(1. + discriminator_fake))
            discriminator_loss = (self.discriminator_loss_factor * 0.5 *
                                  (d_loss_real + d_loss_fake))

        vqgan_gradients = tape.gradient(
            vq_loss,
            self.vqgan.trainable_variables,
        )
        self.optimizer_vqgan.apply_gradients(
            zip(
                vqgan_gradients,
                self.vqgan.trainable_variables,
            ))

        discriminator_gradients = tape.gradient(
            discriminator_loss,
            self.discriminator.trainable_variables,
        )
        self.optimizer_vqgan.apply_gradients(
            zip(
                discriminator_gradients,
                self.discriminator.trainable_variables,
            ))

        self.loss_vqgan_tracker.update_state(vq_loss)
        self.loss_discriminator_tracker.update_state(discriminator_loss)

        loss_vqgan_result = self.loss_vqgan_tracker.result()
        loss_discriminator_result = self.loss_discriminator_tracker.result()
        return {
            "loss": loss_vqgan_result + loss_discriminator_result,
            "loss_vqgan": loss_vqgan_result,
            "loss_discriminator": loss_discriminator_result,
        }
